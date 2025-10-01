import streamlit as st
import pandas as pd
import re
from io import StringIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import plotly.express as px

nltk.download('vader_lexicon')

# Parse WhatsApp chat export txt into a dataframe
def parse_whatsapp(txt):
    pattern = r"(\d{1,2}/\d{1,2}/\d{2,4}) (晚上|早上)?(\d{1,2}:\d{2}) - (.*?): (.*)"
    rows = []
    for line in txt.split('\n'):
        m = re.match(pattern, line)
        if m:
            date_str = m.group(1)
            meridian = m.group(2)  # '晚上' or '早上' or None
            time_str = m.group(3)
            sender = m.group(4)
            message = m.group(5)
            # Convert to 24h time if needed
            hour, minute = map(int, time_str.split(':'))
            if meridian == '晚上' and hour < 12:
                hour += 12
            dt_str = f"{date_str} {hour}:{minute:02d}"
            dt = pd.to_datetime(dt_str, errors='coerce', dayfirst=False)
            if pd.notna(dt):
                rows.append({'datetime': dt, 'sender': sender, 'message': message})
    return pd.DataFrame(rows)

st.title("WhatsApp Chat Analyzer")

uploaded = st.file_uploader("Upload WhatsApp chat text file (.txt)", type=['txt'])
if uploaded:
    txt = StringIO(uploaded.getvalue().decode('utf-8')).read()
    df = parse_whatsapp(txt)
    
    if df.empty:
        st.error("Could not parse the chat file. Make sure it’s a WhatsApp exported txt file.")
        st.stop()
        
    st.success(f"Loaded {len(df)} messages")

    min_date = df['datetime'].dt.date.min()
    max_date = df['datetime'].dt.date.max()

    start_date, end_date = st.slider(
        "Select date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    df_filtered = df[(df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date <= end_date)]

    st.write(f"Showing {len(df_filtered)} messages between {start_date} and {end_date}")
    
    #1 Word Cloud
    text = " ".join(df_filtered['message'].dropna())
    text = text.replace("媒體已略去", "")
    if text.strip():
        wc = WordCloud(font_path='NotoSansTC-Thin.ttf',width=1000, height=700, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    #2 Sentiment Intensity analysis
    #sia = SentimentIntensityAnalyzer()
    #df_filtered['sentiment'] = df_filtered['message'].apply(lambda x: sia.polarity_scores(x)['compound'])

    #sentiment_counts = df_filtered['sentiment'].apply(
    #    lambda s: 'Positive' if s>0.05 else ('Negative' if s<-0.05 else 'Neutral')
    #).value_counts()

    #st.bar_chart(sentiment_counts)
    #st.write(f"Avg sentiment: {df_filtered['sentiment'].mean():.3f}")

    #3 Top 50 frequent words/phrases
    # After filtering df_filtered with selected date range
    messages = df_filtered['message'].dropna().str.lower().str.replace("媒體已略去", "").tolist()

    # Simple tokenization: split by non-word chars, filter empty
    words = []
    for msg in messages:
        words.extend(re.findall(r'\b\w+\b', msg))

    # Count frequencies
    counter = Counter(words)
    top50 = counter.most_common(50)

    # Prepare data for plotting
    df_top50 = pd.DataFrame(top50, columns=['word', 'count'])

    # Plot horizontal bar chart with Plotly
    fig_freq = px.bar(df_top50.sort_values('count'), 
                    x='count', y='word',
                    orientation='h',
                    title='Top 50 Frequent Words/Phrases',
                    labels={'count': 'Frequency', 'word': 'Word/Phrase'})
    fig_freq.update_layout(
    height=1200,                            # taller figure to fit more words
    margin=dict(l=150, r=30, t=50, b=50),   # more left margin for long words
    yaxis=dict(tickfont=dict(size=10))      # smaller y-axis font size if needed
    )
    st.plotly_chart(fig_freq, use_container_width=True)

else:
    st.info("Upload a WhatsApp chat.txt file to start analyzing.")
