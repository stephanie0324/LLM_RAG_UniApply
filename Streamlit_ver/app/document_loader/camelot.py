import camelot
from bs4 import BeautifulSoup
import markdownify


def get_camelot_tables(file_path, convert_to_markdown=True):
    """
    使用 camelot 來擷取有明顯框線的 PDF 表格，僅擷取物件格式的表格，不支援從圖片中擷取。
    
    :param file_path: str, PDF 檔案路徑
    :param convert_to_markdown: bool, 轉換成 makrdown 格式
    :return: dict<str, list<str>>, 頁碼對應到的 tables
    """
    
    def _html_to_markdown(html_content):
        """
        將 HTML 內容轉換為 Markdown 格式。
        
        :param html_content: str, HTML 格式的內容
        :return: str, Markdown 格式的內容
        """
        markdown_content = markdownify.markdownify(html_content, heading_style="ATX")
        return markdown_content
    
    def _clean_empty_rows(html_content):
        """
        清理 HTML 表格中的空白列
        """
        # 解析 HTML 內容
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 找到所有表格
        tables = soup.find_all('table')
        
        for table in tables:
            # 找到所有行
            rows = table.find_all('tr')
            for row in rows:
                # 找到所有單元格
                cells = row.find_all('td')
                # 如果所有單元格都是空白，刪除該行
                if all(cell.get_text(strip=True) == '' for cell in cells):
                    row.decompose()
        
        # 返回清理後的 HTML 內容
        return str(soup)

    def _clean_empty_columns(html_content):
        """
        清理 HTML 表格中的空白欄
        """
        
        # 解析 HTML 內容
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 找到所有表格
        tables = soup.find_all('table')
        
        for table in tables:
            # 找到所有行
            rows = table.find_all('tr')
            
            # 假設所有行的列數相同
            if not rows:
                continue
            column_count = len(rows[0].find_all('td'))
            
            # 構建列索引的空白檢查列表
            empty_columns = [True] * column_count
            
            # 檢查每一列是否都是空白
            for row in rows:
                cells = row.find_all('td')
                for i, cell in enumerate(cells):
                    if cell.get_text(strip=True) != '':
                        empty_columns[i] = False
            
            # 刪除所有空白列
            for row in rows:
                cells = row.find_all('td')
                for i in reversed(range(column_count)):
                    if empty_columns[i]:
                        if len(cells) > i:
                            cells[i].decompose()
        
        # 返回清理後的 HTML 內容
        return str(soup)
    
    # ref: https://camelot-py.readthedocs.io/en/master/user/advanced.html#shift-text-in-spanning-cells
    layout_kwargs = {
        'line_margin': 0.1, # default is 0.5
        'word_margin': 0.05, # default is 0.1
        'char_margin': 0.25, # default is 2.0
    }


    tables = camelot.read_pdf(
        file_path,
        layout_kwargs=layout_kwargs,
        pages='all',
        line_scale=45, # 預設為15, 線的粗細縮放，加大會更容易識別 line ，但也會導致文字被當成 line
        flavor='lattice', # 適用於有表格框線
        copy_text=['h','v'], # 補值
        strip_text='\n',
        parallel=True,
        flag_size=50, # 預設為50, 表格邊界偵測，加大會更容易將周圍當成表格
        row_close_tol=10, # 預設為20, row 合併，加大會更容易和並列
        column_close_tol=10, # 預設為20, col 合併，加大會更容易合併欄
    )
    
    page_to_tables = {}
    for i, table in enumerate(tables):
        # 將表格保存為 HTML 文件
        df = table.df
        df.reset_index(drop=True, inplace=True)
        current_table_page = table.page
        html_content = _clean_empty_columns(_clean_empty_rows(df.to_html(index=False, header=False)))
        current_page_tables = page_to_tables.get(current_table_page, [])
        
        # 有開啟 convert_to_markdown 就轉換成 markdown
        if convert_to_markdown:
            current_page_tables.append(_html_to_markdown(html_content))
        else:
            current_page_tables.append(html_content)
            
        page_to_tables[current_table_page] = current_page_tables
        
    return page_to_tables
            
    
        