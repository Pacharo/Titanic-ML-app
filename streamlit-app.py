
import streamlit as st  # type: ignore
import pandas as pd
import joblib
from datetime import date

# load machine learning model
tree_clf = joblib.load('model_dt.pickle')

### Streamlit app code starts here

st.title('Titanic Survival Prediction')

# --- Data preview ---
# with st.expander('Show sample of Titanic data'):
#     df = pd.read_csv('titanic.csv')   
#     st.dataframe(df.head(20))

st.markdown('**Please provide passenger information:**')

# --- FORM UI ---
with st.sidebar.form("titanic_form"):
    sex = st.selectbox('Sex', ['female', 'male'])
    
    dob = st.date_input(
    "Date of birth",
    min_value=date(1900, 1, 1),
    max_value=date.today()
)

# Calculate age internally
    today = date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    sib_sp = st.number_input('# of siblings / spouses aboard:',
                             min_value=0, max_value=100, value=0)
    pclass = st.selectbox('Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1, 2, 3])
    fare = st.number_input('# of parents / children aboard:',
                           min_value=0, max_value=1000, value=0)

    # submit button
    submitted = st.form_submit_button("Submit")

# --- Processing after submit ---
if submitted:
    st.write("### Submitted Values:")
    st.write(f"**Sex:** {sex}")
    st.write(f"**Age:** {int(age)}")
    st.write(f"**Siblings/Spouses:** {int(sib_sp)}")
    st.write(f"**Ticket Class:** {pclass}")
    st.write(f"**Parents/Children:** {int(fare)}")

    # this is how to dynamically change text
    prediction_state = st.markdown('Calculating...')

    ### Now the inference part starts here (MOVED inside 'if submitted')
    passenger = pd.DataFrame(
        {
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'SibSp': [sib_sp],
            'Fare': [fare]
        }
    )

    # Prediction
    y_pred = tree_clf.predict(passenger)

    # Prepare message
    if y_pred[0] == 0:
        msg = 'This passenger is predicted to be: **died**'
    else:
        msg = 'This passenger is predicted to be: **survived**'

    # Update the prediction text
    prediction_state.markdown(msg)

    # --- Show probability if model supports it ---
    try:
        proba = tree_clf.predict_proba(passenger)  # shape: (1, 2) for binary
        # Find the index of the "survived" class = 1 (more robust than assuming position)
        if hasattr(tree_clf, "classes_"):
            survived_idx = list(tree_clf.classes_).index(1)
        else:
            # Fallback: conventionally the positive class is at index 1
            survived_idx = 1

        survival_prob = proba[0][survived_idx]
        st.markdown(f'The survival probability: **{survival_prob:.2f}**')
        st.progress(min(max(survival_prob, 0.0), 1.0))
    except Exception as e:
        # If the model doesn't support predict_proba or another issue occurs
        st.info("This model does not provide probability estimates.")

# Explaination of the code:

st.markdown("""
    ---
    ### How this prediction works
    The model uses the following passenger features:
    - Ticket class  
    - Sex  
    - **Age (calculated from date of birth)**  
    - Number of siblings or spouses aboard  
    - Fare paid  

    **Limitations:**  
    - This is a simple educational model trained on historical Titanic data.  
    - Accuracy is limited and cannot be applied to real-world predictions.  
    """)
