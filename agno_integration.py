"""
Agno Integration for Advanced File Analysis
This module provides enhanced analysis capabilities using the Agno framework
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import ollama

class AgnoAnalyzer:
    """Advanced analyzer using Agno framework principles for financial data"""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_financial_patterns(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze financial patterns across multiple dataframes using Agno principles
        """
        try:
            analysis_results = {
                'financial_health_scores': {},
                'trend_analysis': {},
                'anomaly_detection': {},
                'cross_file_insights': {},
                'recommendations': []
            }
            
            if not dataframes:
                analysis_results['error'] = 'No dataframes provided'
                return analysis_results
            
            for name, df in dataframes.items():
                try:
                    if df is None or df.empty:
                        continue
                        
                    # Financial health scoring
                    health_score = self._calculate_financial_health(df)
                    analysis_results['financial_health_scores'][name] = health_score
                    
                    # Trend analysis
                    trends = self._analyze_trends(df)
                    analysis_results['trend_analysis'][name] = trends
                    
                    # Anomaly detection
                    anomalies = self._detect_anomalies(df)
                    analysis_results['anomaly_detection'][name] = anomalies
                    
                except Exception as e:
                    # Log error but continue with other files
                    analysis_results['financial_health_scores'][name] = {'error': str(e)}
                    analysis_results['trend_analysis'][name] = {'error': str(e)}
                    analysis_results['anomaly_detection'][name] = {'error': str(e)}
                    continue
            
            # Cross-file insights
            try:
                cross_insights = self._generate_cross_file_insights(dataframes)
                analysis_results['cross_file_insights'] = cross_insights
            except Exception as e:
                analysis_results['cross_file_insights'] = {'error': str(e)}
            
            # Generate recommendations
            try:
                recommendations = self._generate_recommendations(analysis_results)
                analysis_results['recommendations'] = recommendations
            except Exception as e:
                analysis_results['recommendations'] = [f"Error generating recommendations: {str(e)}"]
            
            return analysis_results
            
        except Exception as e:
            return {
                'error': str(e),
                'financial_health_scores': {},
                'trend_analysis': {},
                'anomaly_detection': {},
                'cross_file_insights': {},
                'recommendations': [f"Analysis failed: {str(e)}"]
            }
    
    def _calculate_financial_health(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate financial health score based on available metrics"""
        try:
            health_metrics = {}
            
            # Identify potential financial columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                health_metrics['overall_score'] = 0.5  # Neutral if no numeric data
                return health_metrics
            
            for col in numeric_cols:
                try:
                    col_lower = col.lower()
                    col_data = df[col].dropna()
                    
                    if len(col_data) < 2:
                        continue
                    
                    # Revenue/Sales indicators
                    if any(term in col_lower for term in ['revenue', 'sales', 'income', 'amount']):
                        # Positive trend is good
                        if len(col_data) > 1:
                            values = col_data.values
                            trend = (values[-1] - values[0]) / (abs(values[0]) + 1e-6)
                            health_metrics[f'{col}_trend_score'] = min(max(trend, -1), 1)
                        
                        # Consistency (lower volatility is better)
                        mean_val = col_data.mean()
                        if mean_val != 0:
                            volatility = col_data.std() / abs(mean_val)
                            health_metrics[f'{col}_consistency_score'] = max(0, 1 - min(volatility, 1))
                    
                    # Cost/Expense indicators
                    elif any(term in col_lower for term in ['cost', 'expense', 'expenditure']):
                        # Negative or controlled growth is good
                        if len(col_data) > 1:
                            values = col_data.values
                            trend = (values[-1] - values[0]) / (abs(values[0]) + 1e-6)
                            health_metrics[f'{col}_control_score'] = min(max(-trend, -1), 1)
                    
                    # Ratio indicators
                    elif any(term in col_lower for term in ['margin', 'ratio', 'percentage', 'percent']):
                        # Higher ratios are generally better
                        avg_ratio = col_data.mean()
                        if avg_ratio > 0:
                            health_metrics[f'{col}_level_score'] = min(avg_ratio / 100, 1)
                        else:
                            health_metrics[f'{col}_level_score'] = 0
                
                except Exception as e:
                    # Skip problematic columns
                    continue
            
            # Overall health score
            if health_metrics:
                scores = [v for v in health_metrics.values() if isinstance(v, (int, float))]
                health_metrics['overall_score'] = np.mean(scores) if scores else 0.5
            else:
                health_metrics['overall_score'] = 0.5  # Neutral if no financial indicators found
            
            return health_metrics
            
        except Exception as e:
            return {'overall_score': 0.5, 'error': str(e)}
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in the data"""
        try:
            trends = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                try:
                    col_data = df[col].dropna()
                    if len(col_data) > 2:
                        # Calculate trend direction and strength
                        x = np.arange(len(col_data))
                        y = col_data.values
                        
                        # Simple linear regression using numpy
                        if len(np.unique(y)) > 1:
                            correlation = np.corrcoef(x, y)[0, 1]
                            if np.isnan(correlation):
                                correlation = 0
                        else:
                            correlation = 0
                        
                        trends[col] = {
                            'direction': 'increasing' if correlation > 0.1 else 'decreasing' if correlation < -0.1 else 'stable',
                            'strength': abs(correlation),
                            'recent_change': (y[-1] - y[0]) / (abs(y[0]) + 1e-6) if len(y) > 0 else 0
                        }
                except Exception as e:
                    trends[col] = {
                        'direction': 'unknown',
                        'strength': 0,
                        'recent_change': 0,
                        'error': str(e)
                    }
            
            return trends
            
        except Exception as e:
            return {'error': str(e)}
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Detect anomalies using statistical methods"""
        try:
            anomalies = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                try:
                    col_data = df[col].dropna()
                    if len(col_data) > 3:
                        # IQR-based anomaly detection
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        
                        if IQR > 0:  # Avoid division by zero
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            
                            anomaly_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                            anomaly_indices = df[anomaly_mask].index.tolist()
                            
                            if anomaly_indices:
                                anomalies[col] = anomaly_indices[:10]  # Limit to first 10 anomalies
                except Exception as e:
                    # Skip problematic columns
                    continue
            
            return anomalies
            
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_cross_file_insights(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate insights across multiple files"""
        insights = {
            'common_patterns': [],
            'data_gaps': [],
            'potential_duplicates': [],
            'complementary_data': []
        }
        
        # Find common column patterns
        all_columns = {}
        for name, df in dataframes.items():
            for col in df.columns:
                col_lower = col.lower()
                if col_lower not in all_columns:
                    all_columns[col_lower] = []
                all_columns[col_lower].append((name, col))
        
        # Identify patterns
        for col_pattern, occurrences in all_columns.items():
            if len(occurrences) > 1:
                insights['common_patterns'].append({
                    'pattern': col_pattern,
                    'files': [occ[0] for occ in occurrences],
                    'columns': [occ[1] for occ in occurrences]
                })
        
        # Detect potential data relationships
        file_names = list(dataframes.keys())
        for i in range(len(file_names)):
            for j in range(i + 1, len(file_names)):
                file1, file2 = file_names[i], file_names[j]
                df1, df2 = dataframes[file1], dataframes[file2]
                
                # Check for potential key relationships
                common_cols = set(df1.columns) & set(df2.columns)
                if common_cols:
                    insights['complementary_data'].append({
                        'file1': file1,
                        'file2': file2,
                        'common_columns': list(common_cols),
                        'potential_join_keys': list(common_cols)
                    })
        
        return insights
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Health-based recommendations
        health_scores = analysis_results['financial_health_scores']
        for file_name, scores in health_scores.items():
            overall_score = scores.get('overall_score', 0)
            
            if overall_score < 0.3:
                recommendations.append(f"🚨 {file_name}: Poor financial health indicators detected. Review expense control and revenue optimization.")
            elif overall_score < 0.6:
                recommendations.append(f"⚠️ {file_name}: Moderate financial health. Consider improving operational efficiency.")
            else:
                recommendations.append(f"✅ {file_name}: Good financial health indicators.")
        
        # Trend-based recommendations
        trends = analysis_results['trend_analysis']
        for file_name, file_trends in trends.items():
            for col, trend_info in file_trends.items():
                if 'revenue' in col.lower() and trend_info['direction'] == 'decreasing':
                    recommendations.append(f"📉 {file_name}: Declining revenue trend in {col}. Investigate market factors.")
                elif 'cost' in col.lower() and trend_info['direction'] == 'increasing':
                    recommendations.append(f"📈 {file_name}: Rising costs in {col}. Review cost management strategies.")
        
        # Cross-file recommendations
        cross_insights = analysis_results['cross_file_insights']
        if cross_insights['complementary_data']:
            recommendations.append("🔗 Consider consolidating related files for better data integration and analysis.")
        
        if not recommendations:
            recommendations.append("📊 Data analysis complete. No major issues detected.")
        
        return recommendations

def integrate_agno_analysis(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to integrate Agno analysis with processed file data
    """
    try:
        if not processed_data:
            return {'error': 'No processed data provided'}
            
        analyzer = AgnoAnalyzer()
        
        # Extract dataframes from processed data
        dataframes = {}
        for sheet_name, data in processed_data.items():
            try:
                if isinstance(data, dict) and 'dataframe' in data:
                    df = data['dataframe']
                    if df is not None and not df.empty:
                        dataframes[sheet_name] = df
                else:
                    # Handle case where data might be a DataFrame directly
                    if hasattr(data, 'empty') and not data.empty:
                        dataframes[sheet_name] = data
            except Exception as e:
                # Skip problematic data
                continue
        
        if not dataframes:
            return {'error': 'No valid dataframes found in processed data'}
        
        # Run analysis
        analysis_results = analyzer.analyze_financial_patterns(dataframes)
        
        return analysis_results
        
    except Exception as e:
        return {
            'error': str(e),
            'financial_health_scores': {},
            'trend_analysis': {},
            'anomaly_detection': {},
            'cross_file_insights': {},
            'recommendations': [f"Agno integration failed: {str(e)}"]
        }