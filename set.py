from flask import Flask, request, render_template
from difflib import get_close_matches
import requests
from bs4 import BeautifulSoup as bs
from transformers import pipeline

# Initialize the Flask application
app = Flask(__name__)

# Initialize sentiment analysis pipeline
pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

classo = "sc-eb7bd5f6-0 fYAfXe"
cleaned_company_names = [
    "Reliance", "Tata Consultancy Services", "HDFC Bank", "ICICI Bank", "State Bank of India", "Infosys",
    "Life Insurance of India", "Hindustan Unilever", "ITC", "Larsen & Toubro", "Bajaj Finance", "HCL Technologies",
    "Axis Bank", "Maruti Suzuki India", "Oil and Natural Gas", "Sun Pharmaceutical Industries", "Hindustan Aeronautics",
    "Kotak Mahindra Bank", "NTPC", "Adani Enterprises", "Tata Motors", "Mahindra & Mahindra", "Ultratech Cement",
    "Adani Ports and Special Economic Zone", "Power Grid of India", "Avenue Supermarts", "Coal India", "Asian Paints",
    "Titan Company", "Hindustan Zinc", "Wipro", "Adani Power", "Adani Green Energy", "Siemens", "Indian Railway Finance",
    "Bajaj Auto", "Bajaj Finserv", "Nestle", "Indian Oil", "Bharat Elec", "JSW Steel", "Jio Financial Services",
    "Tata Steel", "Varun Beverages", "DLF", "Trent", "Zomato", "Grasim Industries", "Power Finance", "Apple", "Microsoft",
    "Saudi Aramco", "Alphabet", "Amazon.com", "NVIDIA", "Berkshire Hathaway", "Meta Platforms", "Tesla",
    "Taiwan Semiconductor Manufacturing", "Tencent Holdings", "Samsung Electronics", "UnitedHealth", "Johnson & Johnson",
    "LVMH Moet Hennessy Louis Vuitton", "Visa", "Exxon Mobil", "JPMorgan Chase", "Eli Lilly", "Procter & Gamble",
    "Roche Holding", "Kweichow Moutai", "Nestle", "Walmart", "TSMC", "ASML Holding", "Mastercard", "Pfizer",
    "Bank of America", "AbbVie", "Chevron", "Novo Nordisk", "PepsiCo", "Coca-Cola", "Oracle", "Novartis", "L'Oreal",
    "Thermo Fisher Scientific", "Broadcom", "Costco Wholesale", "Walt Disney", "Merck", "Adobe", "AstraZeneca", "Nike",
    "Shell", "McDonald's", "China Mobile", "Comcast", "Intel", "Cisco Systems", "Toyota Motor", "Danaher", "Verizon Communications",
    "Salesforce", "Abbott Laboratories", "Medtronic", "Bristol-Myers Squibb", "Amgen", "Philip Morris International",
    "Siemens", "Qualcomm", "Sinopec", "HSBC Holdings", "Sanofi", "Texas Instruments", "General Electric", "Alibaba Holding",
    "SAP", "Honeywell International", "Charter Communications", "Raytheon Technologies", "Caterpillar", "Union Pacific",
    "Diageo", "Boeing", "Gilead Sciences", "China Construction Bank", "Deere", "GlaxoSmithKline", "Royal Bank of Canada",
    "3M", "American Express", "UBS", "Sony", "Target", "Colgate-Palmolive", "AIA", "Schneider Electric", "BNP Paribas",
    "Ford Motor", "Shanghai Pudong Development Bank", "TotalEnergies", "Prologis", "BP", "The Goldman Sachs", "China Merchants Bank",
    "Heineken", "Mitsubishi UFJ Financial", "Sumitomo Mitsui Financial", "Allianz", "Volkswagen", "Hitachi", "BJP",
    "Windows","Airlines US","Netflix","Google"
]

def get_bbc_text(url: str):
    """Parse BBC article and return text in list of strings"""
    article = requests.get(url)
    soup = bs(article.content, "html.parser")
    body = soup.find_all("p", {"class": classo})
    return [b.get_text() for b in body]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    url = request.form['link']
    
    if not url:
        return render_template('index.html', error="URL is required")
    
    result = get_bbc_text(url)
    final_list = []
    final_neg = []
    mark = 1

    for text in result:
        abc = set()
        
        for word in text.split():
            matcho = get_close_matches(word, cleaned_company_names, n=1, cutoff=.70)
            if matcho and word[0] == matcho[0][0]:
                abc.add(matcho[0])
        
        res = pipe(text)
        
        if res[0]['label'] != "neutral":
            mark = -1 if res[0]['label'] == "negative" else 1
            
            for i in abc:
                if i not in final_list:
                    final_list.append(i)
                    final_neg.append(res[0]['score'] * mark)
                else:
                    final_neg[final_list.index(i)] += res[0]['score'] * mark
            
            if not abc:
                final_neg = [final_neg[i] + res[0]['score'] * mark for i in range(len(final_neg))]

    final_list = [x.replace("BJP", "ADANI Group") for x in final_list]
    final_list = list(map(lambda x: x.replace("Windows", "Microsoft"), final_list))
    final_list = list(map(lambda x: x.replace("Google", "Alphabet"), final_list))

    finalpos = [final_list[i] for i in range(len(final_list)) if final_neg[i] > 0]
    finalneg = [final_list[i] for i in range(len(final_list)) if final_neg[i] <= 0]

    return render_template('index.html', positive_companies=finalpos, negative_companies=finalneg)

