import pandas as pd
import os

print("=== COMPREHENSIVE DATASET ANALYSIS ===\n")

# Get all files in the Excel directory
excel_dir = "d:/Dev_space/chatbot/Excel"
if os.path.exists(excel_dir):
    files = os.listdir(excel_dir)
    print(f"Files found in Excel directory: {files}\n")
else:
    print("Excel directory not found\n")

# Define file paths
files_to_analyze = {
    "Revenue Data": "d:/Dev_space/chatbot/Excel/Uniform_Product revenue data.xlsx",
    "Salesforce Data": "d:/Dev_space/chatbot/Excel/uniform_salesforce_data.csv",
    "Gainsight Data": "d:/Dev_space/chatbot/Excel/uniform_gainsight_data.csv",
    "Product Usage": "d:/Dev_space/chatbot/Excel/uniform_product_usage_data.csv",
    "Jira Data": "d:/Dev_space/chatbot/Excel/uniform_jira_data.csv"
}

# Analyze each file
for file_name, file_path in files_to_analyze.items():
    print(f"=== {file_name.upper()} ===")
    
    try:
        # Load the file
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Show data types
        print("\nData Types:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")
        
        # Show sample data
        print("\nSample Data (first 3 rows):")
        print(df.head(3).to_string(index=False))
        
        # Look for key identifier columns
        key_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['customer', 'id', 'product', 'user']):
                key_columns.append(col)
        
        if key_columns:
            print(f"\nKey Identifier Columns: {key_columns}")
            
            # Show unique values for key columns (limited)
            for col in key_columns:
                unique_count = df[col].nunique()
                print(f"  {col}: {unique_count} unique values")
                if unique_count <= 10:
                    print(f"    Values: {df[col].unique().tolist()}")
                else:
                    print(f"    Sample values: {df[col].unique()[:5].tolist()}...")
        
        # Show numeric columns statistics
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            print(f"\nNumeric Columns Statistics:")
            for col in numeric_cols:
                print(f"  {col}:")
                print(f"    Min: {df[col].min()}")
                print(f"    Max: {df[col].max()}")
                print(f"    Mean: {df[col].mean():.2f}")
                if 'revenue' in col.lower() or 'amount' in col.lower():
                    print(f"    Total: {df[col].sum():,.2f}")
        
        print("\n" + "="*60 + "\n")
        
    except Exception as e:
        print(f"Error analyzing {file_name}: {str(e)}")
        print("\n" + "="*60 + "\n")

print("=== CROSS-FILE RELATIONSHIP ANALYSIS ===\n")

# Try to identify common columns across files
all_columns = {}
for file_name, file_path in files_to_analyze.items():
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        all_columns[file_name] = df.columns.tolist()
    except:
        continue

# Find common column patterns
common_patterns = {}
for pattern in ['customer', 'product', 'id', 'date', 'revenue', 'user']:
    matching_files = []
    for file_name, columns in all_columns.items():
        matching_cols = [col for col in columns if pattern.lower() in col.lower()]
        if matching_cols:
            matching_files.append(f"{file_name}: {matching_cols}")
    if matching_files:
        common_patterns[pattern] = matching_files

print("Common column patterns across files:")
for pattern, files in common_patterns.items():
    print(f"\n{pattern.upper()} related columns:")
    for file_info in files:
        print(f"  {file_info}")

print("\n=== POTENTIAL QUERY IMPROVEMENTS ===\n")

query_suggestions = [
    "1. REVENUE ANALYSIS:",
    "   - 'What is the total revenue by customer?'",
    "   - 'Which products generate the most revenue?'",
    "   - 'Show customers with revenue over $50,000'",
    "   - 'Compare monthly vs annual revenue trends'",
    "",
    "2. CUSTOMER HEALTH & SUPPORT:",
    "   - 'Which customers have high support tickets and low health scores?'",
    "   - 'Show customers at risk of churn with their revenue impact'",
    "   - 'Correlation between CSAT scores and renewal likelihood'",
    "",
    "3. PRODUCT USAGE INSIGHTS:",
    "   - 'Which customers have high usage but low revenue?'",
    "   - 'Product adoption rates by customer segment'",
    "   - 'API usage patterns and revenue correlation'",
    "",
    "4. CROSS-FUNCTIONAL ANALYSIS:",
    "   - 'Customer 360 view: revenue + health + support + usage'",
    "   - 'Risk analysis: combine churn risk with revenue data'",
    "   - 'Product performance: usage + revenue + support tickets'",
    "",
    "5. BUSINESS INTELLIGENCE QUERIES:",
    "   - 'Top 10 customers by multiple metrics'",
    "   - 'Revenue at risk from unhealthy customers'",
    "   - 'Product portfolio performance analysis'"
]

for suggestion in query_suggestions:
    print(suggestion)

print("\n" + "="*60)
print("Analysis Complete!")