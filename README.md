# London House Price Spatial Hedonic Modelling
Report on the effect school types have on housing prices in London.

## INTRODUCTION

Quality education is a valuable commodity, particularly where places are limited and demand is high. Several organisations and companies  offer services, often at a cost, to provide detailed information on school attributes and availability throughout the UK, to help parents decide on locations where they are most likely to secure a place in a popular or oversubscribed school.  The prevalence of these services would certainly indicate a strong level of importance placed on proximity to good quality free schools.  London is estimated to need an additional 4,800 secondary school places over the next decade, with growth expected in all boroughs, and competition is strong. The question is, how much are they willing to pay, and is this factor significant when compared the more traditional variables used to estimate house prices?  This report examines the impact of proximity to high performing schools on house prices in the Greater London area in 2016.  
  
The study commences with a background discussion on previous research papers, their scope, methodology and conclusions, with most papers indicating that house prices correlate with school performance data. It then provides an overview of the school system in England, how they are funded, admissions acceptance criteria, and how their performance is measured.  An analysis of the 2016 school and house price data will then be reviewed, followed by the model analysis, and discussion on the results.  The report finds that people are willing to pay more to be closer to private fee-paying and religious schools.


## Jupyter Notebook Files
#### 0_London_House_Prices_Baseline
Primary file with visualisations, spatial diagnostics, OLS regression and spatial Hedonic modelling for baseline case study  
  
**1_Log_Dist_School**       Spatial Hedonic modelling including distance to any schools  
**2_Log_Dist_Of1**          Spatial Hedonic modelling including distance to schools with OFSTED 1 rating  
**3_Log_Dist_Of12**         Spatial Hedonic modelling including distance to schools with OFSTED 1 or 2 rating  
**4_Log_Dist_Prim12**       Spatial Hedonic modelling including distance to Primary schools with OFSTED 1 or 2 rating  
**5_Log_Dist_Sec12**        Spatial Hedonic modelling including distance to Secondary schools with OFSTED 1 or 2 rating  
**6_Log_Dist_Priv**         Spatial Hedonic modelling including distance to Private schools  
**7_Log_Dist_VAS**          Spatial Hedonic modelling including distance to Voluntary Aided (Religious) schools  
**8_Log_Dist_FSM**          Spatial Hedonic modelling including distance to schools with >30% free school meals  
**9_Log_Dist_AcSponsor**    Spatial Hedonic modelling including distance to Academy Sponsor schools  




