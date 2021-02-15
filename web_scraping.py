import requests
import re
import time
import os
import urllib.request
from bs4 import BeautifulSoup

base_urls = {
    'Alexander' : 'http://numismatics.org/search/results?q=authority_facet%3A%22Alexander%20III%20of%20Macedon%22%20AND%20imagesavailable%3Atrue&start=',
    'Antiochus' : 'http://numismatics.org/search/results?q=authority_facet%3A%22Antiochus%20III%20the%20Great%22%20AND%20imagesavailable%3Atrue&start=',
    'Ptolemy' : 'http://numismatics.org/search/results?q=authority_facet%3A%22Ptolemy%20I%20Soter%22%20AND%20imagesavailable%3Atrue&lang=en&start=',
    'Seleucus' : 'http://numismatics.org/search/results?q=authority_facet%3A%22Seleucus%20I%20Nicator%22%20AND%20imagesavailable%3Atrue&lang=en&start='
    }

# Returns up to 20 links to the images of this authority for a given page
def GetLinksForAuthority(base_url, start_index):
    URL = base_url + str(start_index)
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')

    results = soup.find_all('img')

    rev_urls = [];
    obv_urls = [];

    for result in results:
        matches = re.findall(r'(http.*rev.*jpg)',str(result))
        if (matches != []):
            rev_urls.append(matches[0])
        matches = re.findall(r'(http.*obv.*jpg)',str(result))
        if (matches != []):
            obv_urls.append(matches[0])
    
    return (rev_urls, obv_urls)

def GetAllLinksForAuthority(authority_name, count):
    all_rev_urls = []
    all_obv_urls = []
    for index in range(0, count, 20):
        print('please wait')
        time.sleep(.02)
        urls = GetLinksForAuthority(base_urls[authority_name], index)
        if len(urls[0]) == 0:
            break
        all_rev_urls += urls[0]
        all_obv_urls += urls[1]
    # TODO get obverse links when we need them
    return (all_rev_urls, all_obv_urls)

def DownloadImage(url, authority_name):
    file_name = re.findall(r'.*/(.*)', url)[0]
    print('downloading picture {0} please wait'.format(file_name))
    urllib.request.urlretrieve(
            url, 'images/{0}/{1}'.format(authority_name, file_name))

def DownloadImages(authority_name, count):
    print('Downloading Images for ' + authority_name)
    
    if not os.path.exists('images/' + authority_name):
        os.makedirs('images/' + authority_name)
    
    (all_rev_urls, all_obv_urls) = GetAllLinksForAuthority(authority_name, 4000)
    incr = len(all_rev_urls)/count
    if (incr < 1):
        incr = 1
    counter = 0
    while counter < len(all_rev_urls) and count > 0:
        DownloadImage(all_rev_urls[int(counter)], authority_name)
        DownloadImage(all_obv_urls[int(counter)], authority_name)
        counter += incr
        count -= 1
        time.sleep(.02)

if not os.path.exists('images'):
    os.makedirs('images')

DownloadImages('Alexander', 600)
DownloadImages('Antiochus', 600)
DownloadImages('Ptolemy', 600)
DownloadImages('Seleucus', 600)


    
    # with open(filename.grop(1), 'wb') as f:
    #     if 'http' not in url:
    #         url = '{}{}'.format(site,url)
    #     response = requests.get(url)
    #     f.write(response.content)