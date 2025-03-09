from typing import List
from itertools import groupby
import re
import logging
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from unstructured.documents.elements import Header, Footer
from unstructured.cleaners.core import remove_punctuation,clean,clean_extra_whitespace, group_broken_paragraphs
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf
from langchain.schema import Document
from document_loader.camelot import get_camelot_tables

logger = logging.getLogger(__name__)

class CustomPDFLoader(UnstructuredFileLoader):
    use_camelot_table: bool = False

    def _get_elements(self) -> List:
        def pre_process(file_path):
            raw_pdf_elements = partition_pdf(
                filename= file_path,
                languages=["chi_tra", "eng"],
                strategy='hi_res',
                hi_res_model_name="yolox",
                extract_images_in_pdf=False,
                infer_table_structure=True,
                combine_text_under_n_chars=0
            )
            return raw_pdf_elements

        def pre_process_fast(file_path):
            raw_pdf_elements = partition_pdf(
                filename= file_path,
                languages=["chi_tra", "eng"],
                strategy='fast',
                extract_images_in_pdf=False,
                infer_table_structure=False,
                combine_text_under_n_chars=0
            )
            return raw_pdf_elements

        def filter_element_category(elements, filter_category):
            """
            
            """
            return [ element for element in elements if element.category not in filter_category]

        def is_specify_title(elements):
            """
            暫無操作
            """
            return False

        def element_to_document(elements):
            pass

        def norm_text(text):
            return re.sub(r"\d+", "[INT]", text)

        def element_to_text(elements, use_camelot_table=False, camelot_tables=[]):
            """將elements串成string.
            當use_camelot_table=True時，element.category="Table"的element會被取代為camelot_tables(若可以的話)
            """
            text = ""

            for index, element in enumerate(elements):
                current_page = element.metadata.page_number
                
                if element.category == "Table":
                    # 有開啟使用 camelot table 就 pop 第一個出來使用
                    if use_camelot_table:
                        current_page_camelot_tables = camelot_tables.get(current_page, [])
                        if len(current_page_camelot_tables) > 0:
                            get_first_table = current_page_camelot_tables.pop(0)
                            output_table = get_first_table
                        else:
                            output_table = ''
                            logger.info(f"當前頁面 camelot 沒有辨識出表格: {current_page}")
                    else:
                        # 預設使用 Unstructured 解析出來的 html table
                        output_table = element.metadata.text_as_html
                                                        
                    text += output_table.strip() + "\n\n"
                elif element.category in ["Title", "NarrativeText", "UncategorizedText"]:
                    if not is_specify_title(element):
                        text += element.text + "\n"
                else:
                    if element.text:
                        text += element.text + "\n"

                if (index + 1) < len(elements) and elements[index + 1].category in ["Title", "Image"]:
                    text += "\n"

                # 若當前是該頁最後面了，且 camelot 有表格未放置，就放置於該頁最後
                if use_camelot_table and \
                    (
                        # 下一個element換頁了
                        (
                            (index + 1) < len(elements) and \
                            elements[index + 1].metadata.page_number != current_page
                        ) or \
                        (index + 1) >= len(elements)  # 沒有下一頁了
                    ):
                    # 判斷是否有表格未放置
                    current_page_camelot_tables = camelot_tables.get(current_page, [])
                    if len(current_page_camelot_tables) > 0:
                        logger.info(f"將多出來的 camelot table 置於頁尾: {current_page}")
                        text += "\n" + "\n".join(current_page_camelot_tables) +"\n\n"

            return text

        def element_to_text_by_norm_and_rm_duplicated(elements):
            return '\n\n'.join(
                set(
                    norm_text(e.text)
                    for e in elements
                )
            )

        def get_page_header_and_footer(elements):
            """回傳不包含頁首頁尾的elements.
            """
            # logger.info('ALL_element______:\n' + '\n\n'.join([
            #     element.text
            #     for element in elements
            # ]))

            # 初始化變數
            for element in elements:
                element.metadata.is_page_first_element = False
                element.metadata.is_page_last_element = False
                element.metadata.norm_page_content = ''
                element.metadata.has_same_pattern = False
                element.metadata.is_top_or_bottom_y = False

            # 將每頁第一個element標記is_page_first_element=True
            for _, group in groupby(elements, key=lambda x: x.metadata.page_number):
                next(group).metadata.is_page_first_element = True

            # 將每頁最後一個element標記is_page_last_element=True
            for _, group in groupby(elements[::-1], key=lambda x: x.metadata.page_number):
                next(group).metadata.is_page_last_element = True

            # 針對is_page_first_element=True或is_page_last_element=True的element增加 norm_page_content(正規內容)
            for element in elements:
                if element.metadata.is_page_first_element is True or element.metadata.is_page_last_element is True:
                    # 把連續數字轉為[INT]
                    element.metadata.norm_page_content = norm_text(element.text)

            # 針對is_page_first_element=True的element判斷是否前一頁或後一頁有相同的頁首 norm_page_content(正規內容)，滿足則標記'has_same_pattern'=True
            # 針對is_page_last_element=True的element判斷是否前一頁或後一頁有相同的頁尾 norm_page_content(正規內容)，滿足則標記'has_same_pattern'=True
            for t in ['is_page_first_element', 'is_page_last_element']:
                _elements = [
                    element
                    for element in elements
                    if getattr(element.metadata, t) is True
                ]
                for idx, element in enumerate(_elements):
                    if idx >= 1 and _elements[idx-1].metadata.norm_page_content == element.metadata.norm_page_content:
                        element.metadata.has_same_pattern = True
                    if idx < len(_elements) - 1 and _elements[idx+1].metadata.norm_page_content == element.metadata.norm_page_content:
                        element.metadata.has_same_pattern = True

            # 針對 has_same_pattern=True的element增加 'is_top_or_bottom_y'(是否位於頂端15%或底部15%)
            for element in elements:
                if element.metadata.has_same_pattern is not True:
                    continue

                if element.metadata.is_page_first_element is True:
                    y2 = max([y for x, y in element.metadata.coordinates.points])
                    if y2 / element.metadata.coordinates.to_dict()['layout_height'] <= 0.15:
                        element.metadata.is_top_or_bottom_y = True
                elif element.metadata.is_page_last_element is True:
                    y1 = min([y for x, y in element.metadata.coordinates.points])
                    if y1 / element.metadata.coordinates.to_dict()['layout_height'] >= 0.85:
                        element.metadata.is_top_or_bottom_y = True
                else:
                    raise

            return [
                element
                for element in elements
                if element.metadata.is_top_or_bottom_y is True
            ]

        ##################
        # 主要入口點在這 #
        ##################
        raw_pdf_fast_elements = pre_process_fast(self.file_path)
        footer_or_header_elements = get_page_header_and_footer(raw_pdf_fast_elements)
        footer_or_header_texts = set((e.text, e.metadata.page_number) for e in footer_or_header_elements)
        logger.info('以下element被判斷為頁首頁尾, 待刪除:\n' + '\n\n'.join([
            element.text
            for element in footer_or_header_elements
        ]))

        raw_pdf_elements = pre_process(self.file_path)

        # 過濾被Unstructured標記為Header或Footer的element.
        # NOTE(Y.L.Tsai): 這個會與下方的自定義邏輯衝突，因此先拿掉，只靠下方rule移除頁首頁尾.
        # result_elements = filter_element_category(raw_pdf_elements, ["Header", "Footer"])

        # 用自定義邏輯(正規後判斷頁首頁尾是否有重複pattern)刪除一次頁首頁尾
        result_elements = [e for e in raw_pdf_elements if (e.text, e.metadata.page_number) not in footer_or_header_texts]
        footer_or_header_elements = [e for e in raw_pdf_elements if (e.text, e.metadata.page_number) in footer_or_header_texts]
        logger.info('以下element被判斷為頁首頁尾, 已刪除:\n' + '\n\n'.join([
            e.text
            for e in footer_or_header_elements
        ]))

        # result_elements = get_page_header_and_footer(raw_pdf_elements)

        # result_elements = chunk_by_title(result_elements)
        if self.use_camelot_table:
            camelot_tables = get_camelot_tables(self.file_path)
            logger.info("已取得 camelot_tables")
        else:
            camelot_tables = []
        
        main_content_text = element_to_text(result_elements, self.use_camelot_table, camelot_tables)
        footer_or_header_text = element_to_text_by_norm_and_rm_duplicated(footer_or_header_elements)
        logger.info("return PDF result....")
        return [main_content_text, footer_or_header_text]
