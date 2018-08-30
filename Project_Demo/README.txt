================================================================================================================================

CIS 600 - Principles of Social Media and Data Mining by Prof. Martin Harrison

Term Project- Market Value Product Analysis using Sentiment Analysis
Authors- Amritbani Sondhi, Joy Shalom Soosai Michael, Sachin Basavani, Akshay Karthick Manoharan

================================================================================================================================

Market_Value_Product_Summary_Demo:
---------------------------------
To run the Demo successfully, follow the steps below:
1. Create an account on Plotly and verify the account: https://plot.ly/
	- This is needed to generate interactive plots of Sankey and Choropleth plots
	- The plots will be generated at runtime, and the same will be displayed on the Bokeh Server UI

2. Go to: Your 'Account' -> 'Settings' -> 'API Keys' -> Click on 'Regenerate Key' -> Copy the 'Key'

3. Open the 'Market_Value_Product_Summary_Demo.py' in the current folder

4. Uncomment the first 2 lines of code, and enter your 'Username' and 'API key' on the respective places

5. Open Anaconda Command Terminal, and enter the following commands:
	- cd ".\<Your_Directory_Path>\Market Value Product Analysis\Project_Demo\"
	- bokeh serve Market_Value_Product_Summary_Demo.py
	
================================================================================================================================

Congratulations! Your Bokeh Server is successfully running now!
--------------------------------------------------------------------------------------------------------------------------------
The file 'tokenizedDataframe.pickle' in the current folder already contains the final output of 
'Market_Value_Product_Summary_Sentiment_Analysis.ipynb' from the previous folder

================================================================================================================================

Bokeh UI Interaction:
---------------------
There are three buttons for 3 different analysis for the 2 products:
1. Sankey Analysis
   ---------------
	- on click event, an Interactive Sankey Plot is created at runtime and gets launched in a new tab of the browser
	- The two plots take about 5-7 seconds to properly generate and reflect on the UI too
	- The Bokeh UI shows the image of the plot. Bokeh is interactive in itself for it's own plots and visualizations.
		However, it gets limited when it has to perform intra-library functionalities. Hence, the interactions available through
		Plotly library does not work on the Bokeh Server.
	- On the launched Plotly analysis, if you hover on any part, it will show the information about the incoming and outgoing
		data, at that specific point
	- The two plots for Sankey Analysis for Samsung and Apple are shown one after another, so it can be easily compared
	
2. Word Cloud
   ----------
	- on click event, the word clouds are generated using matplotlib and shown on the Bokeh gridplots accordingly
	- all the six plots for positive, negative and neutral analysis for Samsung and Apple are shown respectively
	
3. Region Analysis
   ---------------
	- On click event, the region analysis is done by creating Choropleth Maps on the fly
	- On the launched Plotly analysis, if you hover on any part, it will show the information about the country name, and 
		it's intuitive polarity for that region. Red- for negative, Gray- for neutral and Green- for Positive.
	- Similar to Sankey Analysis, it launches two new interactions in a new tab with Plotly
	- For this too, it takes 5-7 seconds to properly create and render the plots on the UI
	- For ease of comparison between regions, the two plots for Choropleth Analysis for Samsung and Apple are shown one after 
		another
		
================================================================================================================================