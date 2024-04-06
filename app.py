from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('CES_RANDOM_FOREST_MODEL.pkl')

#กำหนดฟังก์ชัน
@app.route('/', methods=['GET', 'POST'])
def predict_branch():
    prediction_result = None
    if request.method == 'POST':
        # Extract input data from the form
        S_RANK = int(request.form['S_RANK'])
        P_ID = int(request.form['P_ID'])
        R_ID = int(request.form['R_ID'])
        FEE_NAME = int(request.form['FEE_NAME'])
        S_PARENT = int(request.form['S_PARENT'])
        GPA = int(request.form['GPA']) 
        GPA_MATCH = int(request.form['GPA_MATCH'])
        GPA_SCI = int(request.form['GPA_SCI'])

        # Prepare the input data for prediction
        new_data = [[S_RANK, P_ID, R_ID, FEE_NAME, S_PARENT, GPA, GPA_MATCH, GPA_SCI]]

        # Perform prediction
        predicted_label = model.predict(new_data)[0]

        # Map the numeric label to its corresponding string representation
        mapping = {0: 'นิเทศศสาตร์ดิจิทัล', 1: 'ดิจิทัลคอนเทนต์และสื่อ', 2: 'อินเทอร์แอคทีฟ มัลติมีเดีย แอนิเมชันและเกม', 3: 'เทคโนโลยีสารสนเทศและนวัติกรรมดิจิทัล', 4: 'นวัตกรรมสารสนเทศทางการเเพทย์'}
        predicted_branch = mapping[predicted_label]

        # Set the prediction result to be displayed on the website
        prediction_result = f"หลักสูตร {predicted_branch}"

    return render_template('index.html', prediction_result=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
