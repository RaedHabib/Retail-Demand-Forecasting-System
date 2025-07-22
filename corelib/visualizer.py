import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    """Handles exploratory data analysis and visualization"""

    def __init__(self, config):
        self.config = config
        plt.style.use(config['visualization']['style'])
        sns.set_palette(config['visualization']['palette'])

    def create_visualizations(self, df):
        """Generate all EDA visualizations"""
        logger.info("Creating visualizations...")
        self._plot_temporal_trends(df)
        self._plot_weekly_patterns(df)
        self._plot_product_analysis(df)
        self._plot_warehouse_distribution(df)
        self._plot_holiday_impact(df)
        self._plot_correlation_matrix(df)
        self._plot_outliers(df)
        self._plot_decomposition(df)

    def _plot_temporal_trends(self, df):
        """Plot daily sales trends"""
        plt.figure(figsize=self.config['visualization']['figsize'])
        daily_sales = df.groupby('day')['quantity'].sum()
        daily_sales.plot(title='Daily Sales Trend')
        plt.ylabel('Total Quantity Sold')
        plt.xlabel('Date')
        plt.tight_layout()
        plt.show()

    def _plot_weekly_patterns(self, df):
        """Plot weekly sales distribution"""
        plt.figure(figsize=self.config['visualization']['figsize'])
        sns.boxplot(x='day_of_week', y='quantity', data=df, showfliers=False)
        plt.title('Weekly Sales Distribution (Sunday=0 to Saturday=6)')
        plt.xlabel('Day of Week')
        plt.ylabel('Sales Quantity')
        plt.tight_layout()
        plt.show()

    def _plot_product_analysis(self, df):
        """Plot top products by sales volume"""
        plt.figure(figsize=self.config['visualization']['figsize'])
        top_products = df.groupby('product_id')['quantity'].sum().nlargest(10)
        top_products.plot(kind='bar', title='Top 10 Products by Sales Volume')
        plt.xlabel('Product ID')
        plt.ylabel('Total Quantity Sold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def _plot_warehouse_distribution(self, df):
        """Plot sales distribution by warehouse"""
        plt.figure(figsize=self.config['visualization']['figsize'])
        warehouse_dist = df.groupby('warehouse_id')['quantity'].sum()
        warehouse_dist.plot(
            kind='pie',
            autopct='%1.1f%%',
            colors=sns.color_palette('Blues'),
            title='Sales Distribution by Warehouse'
        )
        plt.ylabel('')
        plt.tight_layout()
        plt.show()

    def _plot_holiday_impact(self, df):
        """Plot holiday vs non-holiday sales comparison"""
        plt.figure(figsize=(8, 4))
        holiday_sales = df.groupby('is_holiday')['quantity'].mean()
        holiday_sales.plot(kind='bar', title='Average Sales: Holiday vs Non-Holiday')
        plt.xticks([0, 1], ['Non-Holiday', 'Holiday'], rotation=0)
        plt.ylabel('Average Quantity Sold')
        plt.tight_layout()
        plt.show()

    def _plot_correlation_matrix(self, df):
        """Plot feature correlation matrix"""
        plt.figure(figsize=self.config['visualization']['figsize'])
        corr_matrix = df[['quantity', 'lag1', 'lag7', 'rolling_mean7', 'is_holiday']].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            annot=True,
            mask=mask,
            cmap='Blues',
            vmin=-1,
            vmax=1,
            linewidths=0.5
        )
        plt.title('Feature Correlation Matrix (Lower Triangle)')
        plt.tight_layout()
        plt.show()

    def _plot_outliers(self, df):
        """Plot sales quantity distribution with outliers"""
        plt.figure(figsize=self.config['visualization']['figsize'])
        sns.boxplot(x=df['quantity'], showfliers=True)
        plt.title('Sales Quantity Distribution')
        plt.xlabel('Quantity per Transaction')
        plt.tight_layout()
        plt.show()

    def _plot_decomposition(self, df):
        """Plot time series decomposition"""
        plt.figure(figsize=(12, 8))
        daily_sales = df.groupby('day')['quantity'].sum()
        decomposition = seasonal_decompose(daily_sales, period=7)

        # Create subplots with shared x-axis
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)

        # Observed
        decomposition.observed.plot(ax=ax1, color=self.config['visualization']['palette'][0])
        ax1.set_ylabel('Observed')

        # Trend
        decomposition.trend.plot(ax=ax2, color=self.config['visualization']['palette'][0])
        ax2.set_ylabel('Trend')

        # Seasonal
        decomposition.seasonal.plot(ax=ax3, color=self.config['visualization']['palette'][0])
        ax3.set_ylabel('Seasonal')

        # Residual
        decomposition.resid.plot(ax=ax4, color=self.config['visualization']['palette'][0])
        ax4.set_ylabel('Residual')

        plt.suptitle('Time Series Decomposition (Trend, Seasonality, Residual)')
        plt.tight_layout()
        plt.show()