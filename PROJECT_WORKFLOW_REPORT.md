# Fleet Maintenance Project Report

## 1. Project Goal

This project predicts whether a vehicle may need maintenance soon and then explains what should be done next in simple language.

It combines three ideas:

- Machine Learning to estimate maintenance risk
- Retrieval to fetch the right maintenance rules
- An AI agent step to turn the results into a readable report

## 2. Simple Workflow

The workflow is:

1. The user enters vehicle details in the Streamlit app.
2. The ML model checks the sensor values and predicts the maintenance risk.
3. The system looks up the most relevant maintenance guidelines from the local knowledge base.
4. The AI agent combines the prediction and the guidelines.
5. The app shows a final report with:
   - health summary
   - action plan
   - safety disclaimer

## 3. Input to Output Pipeline

### Step 1: User Input

The user enters:

- vehicle type
- usage hours
- engine temperature
- tire pressure
- oil quality
- battery voltage
- vibration level
- maintenance cost
- brake condition

### Step 2: ML Prediction

The model checks the input values and gives:

- a risk score
- a risk label: `High Risk` or `Safe`
- the top contributing factors

This helps answer:

“Is this vehicle likely to need maintenance soon?”

### Step 3: Guideline Retrieval

The system searches the maintenance manual and pulls the most relevant guideline text.

This helps answer:

“What do the maintenance rules suggest for this kind of vehicle condition?”

### Step 4: Agent Step

The agent works in 3 simple parts:

1. Read the ML prediction
2. Read the retrieved maintenance guidelines
3. Ask the LLM to write one clean final report

So the agent is not magic. It is just a smart pipeline that joins:

- prediction
- knowledge
- explanation

### Step 5: Final Report

The app shows:

- **Health Summary**: short explanation of vehicle condition
- **Action Plan**: what should be done next
- **Disclaimer**: safety note for the operator

## 4. Files After Simplification

The code is now simplified into:

- `app.py` for Streamlit UI
- `fleet_system.py` for the full backend logic

This means the main logic is easy to explain in one place.

## 5. How the Agent Works in Very Simple Language

You can explain it like this:

“First, the model predicts if the vehicle is risky. Then the system reads the maintenance rules. After that, the AI writes a clear report based on both.”

## 6. Why This Project Is Useful

This project helps in:

- reducing sudden vehicle failures
- making maintenance decisions faster
- giving a simple report instead of raw numbers
- helping fleet managers understand what action to take

## 7. End-to-End Summary

In one line:

**Vehicle data goes in -> ML predicts risk -> guidelines are fetched -> AI explains the result -> final maintenance report comes out.**
