import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score






# Load the cleaned dataset
df = pd.read_csv("cleaned_ebay_deals.csv")
df = df.dropna(subset=['price', 'original_price', 'shipping', 'discount_percentage'])
print("Loaded rows:", len(df))

# Convert shipping to numeric values
def convert_shipping(shipping):
    if pd.isna(shipping):
        return 0.0  # Handle NaN values
    shipping = str(shipping).strip()
    if shipping == 'Free shipping':
        return 0.0
    elif shipping == 'Shipping info unavailable':
        return 0.0  # Treat unavailable as free shipping
    elif '$' in shipping:
        try:
            return float(shipping.replace('$', '').replace(',', '').strip())
        except:
            return 0.0
    else:
        try:
            return float(shipping)
        except:
            return 0.0  # Default to free shipping if conversion fails

df['shipping'] = df['shipping'].apply(convert_shipping)
# Create a new column: high_discount (True if discount > 20%)
df['high_discount'] = df['discount_percentage'] > 20

# Display sample rows
print(df[['price', 'original_price', 'discount_percentage', 'high_discount']].head())
# Features (X): price and original price
X = df[['price', 'original_price']]

# Target (y): high_discount (boolean)
y = df['high_discount']
print("Final dataset shape (X):", X.shape)
print("Final labels shape (y):", y.shape)

from sklearn.model_selection import train_test_split

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LogisticRegression

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Detailed performance report
print(classification_report(y_test, y_pred))
import seaborn as sns
import matplotlib.pyplot as plt

# Plot prediction counts
sns.countplot(x=y_pred)
plt.title("Predicted: High Discount Classification")
plt.xlabel("High Discount (True/False)")
plt.ylabel("Number of Products")
plt.show()

plt.figure(figsize=(10,6))
plt.hist(df['discount_percentage'], bins=50)
plt.title('Distribution of Discount Percentage')
plt.xlabel('Discount Percentage')
plt.ylabel('Frequency')
plt.show()

bins = [0, 10, 30, float('inf')]
labels = ['Low', 'Medium', 'High']
df['discount_bin'] = pd.cut(df['discount_percentage'], bins=bins, labels=labels)


bin_counts = df['discount_bin'].value_counts()
print(bin_counts)

min_count = min(bin_counts)
balanced_df = pd.concat([
    df[df['discount_bin'] == 'Low'].sample(min_count),
    df[df['discount_bin'] == 'Medium'].sample(min_count),
    df[df['discount_bin'] == 'High'].sample(min_count)
])


balanced_df = balanced_df.drop(columns=['discount_bin'])


X = balanced_df[['price', 'original_price', 'shipping']]
y = balanced_df['discount_percentage']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")


plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Discount Percentage')
plt.ylabel('Predicted Discount Percentage')
plt.title('Actual vs Predicted Discounts')
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(10,6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()


incomplete_df = df.drop(columns=['discount_percentage']).sample(20)


incomplete_df['predicted_discount'] = model.predict(incomplete_df[['price', 'original_price', 'shipping']])


results_table = incomplete_df[['title', 'price', 'original_price', 'shipping', 'predicted_discount']]
print(results_table.to_markdown(index=False))