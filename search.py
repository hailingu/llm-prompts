import urllib.request, urllib.parse, re
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_a = False
        self.in_snippet = False
        self.results = []
        self.current_title = ""
        self.current_snippet = ""
        
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'a' and 'class' in attrs and 'result__url' in attrs['class']:
            self.in_a = True
        if tag == 'a' and 'class' in attrs and 'result__snippet' in attrs['class']:
            self.in_snippet = True
            
    def handle_endtag(self, tag):
        if tag == 'a':
            if self.in_a:
                self.in_a = False
            if self.in_snippet:
                self.in_snippet = False
                self.results.append({'title': self.current_title.strip(), 'snippet': self.current_snippet.strip()})
                self.current_title = ""
                self.current_snippet = ""
                
    def handle_data(self, data):
        if self.in_a:
            self.current_title += data
        if self.in_snippet:
            self.current_snippet += data

def search(query):
    req = urllib.request.Request('https://html.duckduckgo.com/html/?q=' + urllib.parse.quote(query), headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    html = urllib.request.urlopen(req).read().decode('utf-8')
    parser = MyHTMLParser()
    parser.feed(html)
    for r in parser.results[:5]:
        print("TITLE:", r['title'])
        print("SNIPPET:", r['snippet'])
        print("---")

search('95% confidence rule AI agents LLMs')
search('95% confidence rule AI')
