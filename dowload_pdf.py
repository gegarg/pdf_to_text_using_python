## This file was created by Geeta Garg, Indiana University (email: gegarg@iu.edu) to download pdf transcripts from Federal Reserve's website for a summer project
## from the website "https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm"

########################################   Import Required Libraries    #############################
from urllib.request import urlopen
from bs4 import BeautifulSoup
from time import sleep
import os
import sys
import argparse

#######################################################################################################
## Create the URLs from where the files need to be downloaded
#######################################################################################################
Years = range(1975, 1988) # range of years

# define the function to create URLs
def get_url(dates):
    url_list_final = []   # list to save URLs
    #  url_fin = []
    for start_date in dates:
        url = {
            'first': "https://www.federalreserve.gov/monetarypolicy/fomchistorical",
            'Date' : str(start_date)
        }
        url_fin =  url['first'] + url['Date'] + ".htm"
        url_list_final.append(url_fin)
    return url_list_final

url_final = get_url(Years)
print("url_final", url_final)

#for urls in url_final:
#    print("urls_first", urls)

######################################################################################################
# Once the URLs are created in the above step, use this code to download the pdf files and save them
# This part of the code creates URLS as well as the name (refer: tail) by which the file will be saved
######################################################################################################

def get_urls(url):
    urls = []

    for url_each in url:
        # print("url_each", url_each)
        request = urlopen(url_each, None)
        soup = BeautifulSoup(request.read(), "html.parser") # Run the document through Beautiful Soup to convert HTML document into a Python object
        #print("soup", soup.prettify())
        base_url = os.path.dirname(url_each)
        # print("base_url", base_url)
        anchor = soup.findAll("div", {"class": "col-xs-12 col-md-6"}) # locate the links to all pdf files in the URL
        #print("anchor", anchor)
        for items in anchor:
            items1 = items.find_all('a', href=True) #Going inside links --> get all 'a' tags
            #   print("items1", items1)
            for link in items1:
                link1 = link.get('href') # extract the URL to pdfs within 'a' tags
                #   print("link1", link1)
                req = link['href']
                tail = os.path.basename(req)
                head = os.path.dirname(req)
                # print("head", head)
                Segments = head.rpartition('/')
                #print("segments2", Segments[2])
                full_url = "{}/{}/{}".format(base_url, Segments[2], tail)   # create the full URL to each pdf
                urls.append((tail, full_url))
    # print("urls", urls)
    return urls

######################################################################################################
#This part downloads the pdf files using the URLs created earlier and saves them in the present working
#directory using the names (tails) of the files
######################################################################################################

def download(urls, path):
    old_dir = os.getcwd()    ## Returns the current working directory
    os.chdir(path)        ## changing the directory from cwd to the suggested path
    for tail, url in urls:
        #print("url", url)
        if os.path.isfile(tail):
            print(tail, "already exists, skipping...")
            continue
        try:
            request = urlopen(url)
            res = request.read()
            with open(tail, 'wb') as pdf:
                pdf.write(res)
            print("Downloaded", tail)
        except Exception as e:
            print("Download failed", tail, ", because of", e)
    os.chdir(old_dir)

######################################################################################################
# save the files
download(get_urls(url_final), '/Users/geetagarg/newshist/fed_downloads/fed_files')
######################################################################################################

if __name__ == "__main__":
    main()





