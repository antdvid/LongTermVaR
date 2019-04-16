import xml.etree.ElementTree as ET
from datetime import datetime
import pyodbc
import os.path
import datetime as dt
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def get_security_from_lookups(lookups, date):
    sp_template = "exec dbo.IC_DataRetrieval_SingleSecurity @SecIdType = '', @SecId = '', @Lookup = '%s', @AsOfDate='%s'"
    sp = [sp_template % (lkup, date) for lkup in lookups]
    return get_xml_from_stored_procedure(sp, lookups)

def get_xml_from_stored_procedure(sps, filenames):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')  # Last I checked this was necessary.
    browser = webdriver.Chrome(chrome_options=options)
    browser.get('http://calypso.axiomainc.com/ClientIntercom_PROD_MAC_PRODDB/GetDataFromDb.aspx')

    for sp, fname in zip(sps, filenames):
        print("Executing %s"%sp)
        textinput = browser.find_element_by_xpath('//*[@id="SqlTxtBox_ForDisplay"]')
        textinput.clear()
        textinput.send_keys(sp)

        button = browser.find_element_by_xpath("/html/body/div/input[2]")
        button.click()

        text = browser.find_element_by_xpath('//*[@id="OutputTxtArea"]').text
        with open("%s.xml"%fname, "w") as text_file:
            print(text, file=text_file)
    browser.close()

def get_stock_observation_from_axiomadataid(axioma_data_id):
    cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                          "Server=devtest-mac-db-ny;"
                          "Database=MarketData;"
                          "UID=InterCom;"
                          "PWD=inter2011com")
    cursor = cnxn.cursor()
    cursor.execute("exec dbo.IC_StockTS @AxiomadataIds = '%s', @AsOfDate='2019/03/04'"%(axioma_data_id))

    dates = []
    observations = []
    for row in cursor:
        dates.append(row[1])
        observations.append(row[2])
    return dates, observations


def get_data_from_dataid(axid):
    cache_fname = "./cache_%s.txt" % (axid)

    # if file not exist or empty size, recreate the file
    if (not os.path.isfile(cache_fname)) or os.stat(cache_fname).st_size == 0:
        dates, obs = get_stock_observation_from_axiomadataid(axid)
        f = open(cache_fname, "w")
        for kvp in zip(dates, obs):
            msg = "%s  %s\n"%(kvp[0], kvp[1])
            f.writelines(msg)
            print(msg)
        f.close()

    # load file to memory using numpy format
    ds = np.loadtxt(cache_fname, usecols=[0], dtype="str")
    obs_np = np.loadtxt(cache_fname, usecols=[1])
    dates_np = [dt.datetime.strptime(d, '%Y-%m-%d') for d in ds]
    return dates_np, obs_np


def get_observation_from_xml(filename):
    tree = ET.parse(filename)

    root = tree.getroot()

    observations = []
    dates = []
    # date example: 01/16/2001 00:00:00
    date_format = "%m/%d/%Y %H:%M:%S"
    for item in root.findall('TermsAndConditions/EquityObservations'):
        for child in item[2:]:
            text = child.text
            text =text.split(',')
            date = datetime.strptime(text[1], date_format)
            dates.append(date)
            observations.append(float(text[2]))
    return dates, observations

if __name__ == "__main__":
    #d, ob = get_observation_from_xml("ibm.xml")
    #get_stock_observation_from_axiomadataid(axioma_data_id='1692')
    axiomadataid = ['11', '12', '13', '14', '15']
    lookups = ['axiomadataid='+id for id in axiomadataid]
    date = '2018/09/21'
    get_security_from_lookups(lookups, date)