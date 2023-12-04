import gradio as gr
from fastai.imports import *
from fastai.tabular.all import *

# Load trained model
learn = load_learner('model.pkl')

def predict_resale_price(town, flat_type, block, street_name, flat_model, storey_range, floor_area_sqm, remaining_lease_years):
    # Convert string inputs to their respective types to match the trained model's input
    remaining_lease_years = int(remaining_lease_years)
    floor_area_sqm = float(floor_area_sqm)
    
    # Create a DataFrame for the input data
    input_data = pd.DataFrame([[town, flat_type, block, street_name, storey_range, flat_model, floor_area_sqm, remaining_lease_years]], 
                              columns=['town', 'flat_type', 'block', 'street_name', 'storey_range', 'flat_model', 'floor_area_sqm', 'remaining_lease_years'])
    
    # Make predictions
    pred = learn.predict(input_data.iloc[0])[0]
    
    # extract the resale price from the prediction
    resale_price = pred['resale_price'].item()
    
    formatted_resale_price = f"SGD${resale_price:,.2f}"  # Formats the price with comma separators and 2 decimal places
    
    return formatted_resale_price

# Define inputs and outputs for Gradio interface
inputs = [
    gr.Dropdown(choices=['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
       'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
       'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
       'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
       'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
       'TOA PAYOH', 'WOODLANDS', 'YISHUN'], label='Town'),
    gr.Dropdown(choices=['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'MULTI-GENERATION', 'EXECUTIVE'], label='Flat Type'),
    gr.Textbox(label='Block', placeholder='e.g., 155'),
    gr.Textbox(label='Street Name', placeholder='e.g., Ang Mo Kio Ave 3'),
    gr.Dropdown(choices=['Improved', 'New Generation', 'DBSS', 'Standard', 'Apartment',
       'Simplified', 'Model A', 'Premium Apartment', 'Adjoined flat',
       'Model A-Maisonette', 'Maisonette', 'Type S1', 'Type S2',
       'Model A2', 'Terrace', 'Improved-Maisonette', 'Premium Maisonette',
       'Multi Generation', 'Premium Apartment Loft', '2-room'], label = 'Flat Model'),
    gr.Dropdown(choices=['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15',
       '16 TO 18','19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30','31 TO 33',
       '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45','46 TO 48',
       '49 TO 51',], label='Storey Range'),
    gr.Slider(minimum=0, maximum=250, label='Floor Area (sqm)'),
    gr.Dropdown(choices=[str(year) for year in range(0, 99)], label='Remaining Lease Years'), 
]

title = "HDB Resale Price Prediction"
description = "An HDB Resale Price Predictor trained on transaction records from 2017-2020. This tool estimates the resale price of HDB flats in Singapore based on factors like town, flat type, block number, street name, storey range, flat model, floor area, and lease details. Created as a demo for Gradio and HuggingFace Spaces."


output = gr.Textbox(label='HDB Resale Price Prediction')

# Create the Gradio app
iface = gr.Interface(fn=predict_resale_price, inputs=inputs, outputs=output, title=title, description=description)
# Launch the app
iface.launch()