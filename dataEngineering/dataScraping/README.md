# googleNews & glassDoor Data Scraping

Basic implementation of data scraping methods using requests and BeautifulSoup.
For downloading dependencies:

`pip install requests`

`pip install bs4`

Another way to scrape data from webpages is with wget (https://www.linuxjournal.com/content/downloading-entire-web-site-wget):

`wget \
     --recursive \
     --no-clobber \
     --page-requisites \
     --html-extension \
     --convert-links \
     --restrict-file-names=windows \
     --domains website.org \
     --no-parent \
         www.website.com`
         
