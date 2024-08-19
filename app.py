from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from difflib import get_close_matches
import requests
from bs4 import BeautifulSoup as bs
from transformers import pipeline
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=False, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    first_name = db.Column(db.String(150), nullable=False)
    last_name = db.Column(db.String(150), nullable=False)
    


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

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
    try:
        article = requests.get(url)
        article.raise_for_status()  # Raise HTTPError for bad responses
    except requests.RequestException as e:
        return [f"Error fetching the article: {str(e)}"]
    soup = bs(article.content, "html.parser")
    body = soup.find_all("p", {"class": classo})
    return [b.get_text() for b in body]

@app.route('/')
def index():
    if 'explore_landing' not in session:
        if current_user.is_authenticated:
            # Logout the user if accessing the landing page directly when logged in
            logout_user()
            flash("You have been logged out to access the landing page. Please log in again if needed.", "info")
            return redirect(url_for('login'))
    else:
        session.pop('explore_landing', None)
    return render_template('index.html')

@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    url = request.form['link']
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    result = get_bbc_text(url)
    final_list = []
    final_neg = []
    mark = 1

    for text in result:
        abc = set()
        
        for word in text.split():
            matcho = get_close_matches(word, cleaned_company_names, n=1, cutoff=.70)
            if matcho:
                if (word[0]==matcho[0][0]):
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
    final_list=list(map( lambda x: x.replace("Windows","Microsoft"),final_list))
    final_list=list(map( lambda x: x.replace("Google","Alphabet"),final_list))
    finalpos = [final_list[i] for i in range(len(final_list)) if final_neg[i] > 0]
    finalneg = [final_list[i] for i in range(len(final_list)) if final_neg[i] <= 0]

    return render_template('home.html', positive_companies=finalpos, negative_companies=finalneg)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash('Username and password are required', 'warning')
            return redirect(url_for('login'))
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            session['explore_landing'] = True  # Allow exploring the landing page after login
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        first=request.form.get('first_name')
        last=request.form.get('last_name')
        if not username or not password:
            flash('Username and password are required', 'warning')
            return redirect(url_for('register'))
        """if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        """
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password, first_name=first ,last_name=last)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error creating account: {str(e)}', 'danger')
    return render_template('register.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('website_experience')
        email = request.form.get('relevance')
        message = request.form.get('message')
        if not name or not email or not message:
            flash('All fields are required', 'warning')
            return redirect(url_for('contact'))
        # Add contact form logic here (e.g., send an email or store the message)
        flash('Message sent successfully!', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# # Error handling
# @app.errorhandler(HTTPException)
# def handle_http_exception(e):
#     return render_template('login.html')

# @app.errorhandler(Exception)
# def handle_exception(e):
#     app.logger.error(f"An error occurred: {str(e)}")
#     return render_template('login.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
