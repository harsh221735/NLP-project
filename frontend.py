import streamlit as st
import requests
import pandas as pd

# Backend URL (Change this if Flask runs elsewhere)
BACKEND_URL = "http://127.0.0.1:5002/sentiment"

st.set_page_config(page_title="Reddit Car Sentiment Analyzer", page_icon="🚗", layout="centered")

st.title("🚗 Car Sentiment Analyzer (Reddit Data)")
st.write("Analyze public sentiment about any car using Reddit comments.")

# Input field for car name
car_name = st.text_input("Enter a car name (e.g., Maruti Swift, Tata Nexon, Hyundai Creta):")

if st.button("🔍 Analyze Sentiment"):
    if not car_name.strip():
        st.warning("Please enter a valid car name.")
    else:
        with st.spinner("Fetching Reddit comments and analyzing sentiment..."):
            try:
                # Send POST request to backend
                response = requests.post(
                    BACKEND_URL,
                    json={"car_name": car_name},
                    timeout=120
                )

                if response.status_code == 200:
                    data = response.json()

                    if "message" in data and data["message"] == "No comments found":
                        st.error(f"No Reddit comments found for '{car_name}'. Try another car.")
                    else:
                        st.success(f"Sentiment analysis complete for '{data['car_name']}'!")

                        # Display results
                        st.metric("Total Comments", data["total_comments"])
                        st.metric("Average Sentiment Score", round(data["avg_sentiment"], 2))

                        # Convert sentiment labels
                        sentiment_map = {
                            1: "😡 Very Negative",
                            2: "🙁 Negative",
                            3: "😐 Neutral",
                            4: "🙂 Positive",
                            5: "😁 Very Positive"
                        }

                        df = pd.DataFrame([
                            {"Sentiment": sentiment_map.get(int(k), k), "Count": v}
                            for k, v in data["count"].items()
                        ])

                        st.subheader("📊 Sentiment Distribution")
                        st.bar_chart(df.set_index("Sentiment"))

                else:
                    st.error(f"Backend error: {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to backend: {e}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and HuggingFace Transformers")
