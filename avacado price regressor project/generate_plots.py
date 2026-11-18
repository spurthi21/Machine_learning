import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print('Loading data...')
df = pd.read_csv('avocado.csv')

# Prepare helper function
def get_average_between(df, col1, col2):
    vals = []
    groups = list(df[col1].unique())
    for g in groups:
        sub = df[df[col1] == g]
        vals.append(sub[col2].mean())
    res = pd.DataFrame({'column1': groups, 'column2': vals})
    res = res.sort_values('column2', ascending=False).reset_index(drop=True)
    return res

# Create images directory
os.makedirs('backend/static/images', exist_ok=True)

# Plot 1: Price by region
print('Generating plot 1...')
price_by_region = get_average_between(df, 'region', 'AveragePrice')
plt.figure(figsize=(14, 5))
sns.barplot(x='column1', y='column2', data=price_by_region, palette='rocket')
plt.xticks(rotation=90)
plt.title('Average Price by Region')
plt.ylabel('Average Price ($)')
plt.xlabel('Region')
plt.tight_layout()
plt.savefig('backend/static/images/price_by_region.png', dpi=100)
plt.close()

# Plot 2: Production by region
print('Generating plot 2...')
volume_by_region = get_average_between(df, 'region', 'Total Volume')
plt.figure(figsize=(14, 5))
sns.barplot(x='column1', y='column2', data=volume_by_region, palette='viridis')
plt.xticks(rotation=90)
plt.title('Average Total Volume by Region')
plt.ylabel('Total Volume')
plt.xlabel('Region')
plt.tight_layout()
plt.savefig('backend/static/images/production_by_region.png', dpi=100)
plt.close()

# Plot 3: Price by year
print('Generating plot 3...')
price_by_year = get_average_between(df, 'year', 'AveragePrice')
plt.figure(figsize=(10, 5))
sns.barplot(x='column1', y='column2', data=price_by_year.sort_values('column1'), palette='magma')
plt.title('Average Price by Year')
plt.ylabel('Average Price ($)')
plt.xlabel('Year')
plt.tight_layout()
plt.savefig('backend/static/images/price_by_year.png', dpi=100)
plt.close()

# Plot 4: Volume by year
print('Generating plot 4...')
volume_by_year = get_average_between(df, 'year', 'Total Volume')
plt.figure(figsize=(10, 5))
sns.barplot(x='column1', y='column2', data=volume_by_year.sort_values('column1'), palette='cubehelix')
plt.title('Average Total Volume by Year')
plt.ylabel('Total Volume')
plt.xlabel('Year')
plt.tight_layout()
plt.savefig('backend/static/images/volume_by_year.png', dpi=100)
plt.close()

print('All plots generated successfully!')
import glob
files = glob.glob('backend/static/images/*.png')
for f in files:
    size = os.path.getsize(f)
    print(f'  - {os.path.basename(f)}: {size} bytes')
