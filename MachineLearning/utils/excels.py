from openpyxl import load_workbook
from pandas import DataFrame, ExcelFile, ExcelWriter, read_excel

def get_sheet_names(excelfile):
    return (ExcelFile(excelfile)).sheet_names


def read_excels(excelfile, **args):
    if args:
        data = read_excel_range(excelfile, **args)
    else:
        data = read_excel_sheet1(excelfile)

    if data.shape == (1,1):
        return data[0,0]
    elif (data.shape)[0] == 1:
        return data[0]
    else:
        return data


def read_excel_range(excelfile,sheetname="Sheet1",startrow=1,endrow=1,startcol=1,endcol=1):
    values=(read_excel(excelfile, sheetname,header=None)).values;
    return values[startrow-1:endrow,startcol-1:endcol]


def read_excel_sheet1(excelfile):
    return (read_excel(excelfile)).values


def write_excel_data(x,excelfile,sheetname,startrow,startcol):
    df=DataFrame(x)
    book = load_workbook(excelfile)
    writer = ExcelWriter(excelfile, engine='openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df.to_excel(writer, sheet_name=sheetname,startrow=startrow-1, startcol=startcol-1, header=False, index=False)
    writer.save()
    writer.close()


