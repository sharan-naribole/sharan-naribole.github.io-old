---
layout: post
title:  "H-1B Visa Petitions Data Analysis using R (Part I): Data Wrangling"
date:   2017-02-24 12:00:00 -0600
comments: true
---

*This post is part of a series of articles on exploration of H-1B visa petitions public dataset using R language.*

[Part 1: Data Wrangling][h1b-part-I]

[Part 2: Data Analysis][h1b-part-II]

[Part 3: Shiny Web App][h1b-part-III]

[Part 4: Kaggle Open Data][h1b-part-IV]

## Part I: Data Wrangling

*Note: This is a re-post of my blog contributed to* **[NYC Data Science Academy][nyc_dsa]**.

I am excited to begin this blog series on exploring publicly available data on H-1B Visa petitions. In the first part of this series, I will go through the various transformations performed on the raw data set before it is ready for interpretation.

The code used in this blog can be found on [GitHub][github].

<h2> H-1B Visa Introduction </h2>

The H-1B is an employment-based, non-immigrant visa category for temporary foreign workers in the United States. For a foreign national to apply for H1-B visa, an US employer must offer a job and petition for H-1B visa with the US immigration department. This is the most common visa status applied for and held by international students once they complete college/ higher education (Masters, PhD) and work in a full-time position. The Office of Foreign Labor Certification (OFLC) generates [disclosure data][oflc-data] that is useful information about the immigration programs including the H1-B visa.

An important note mentioned on the source website regarding the data:

> Employer-specific case information that appears on this website (and FLCDataCenter.com) was provided to OFLC by employers who submitted foreign labor certification applications. These are not employer responses to the Bureau of Labor Statistics' Occupational Employment Statistics (BLS-OES) survey. The BLS-OES program provides estimates used to assist in setting the wage levels in the FLC wage library. However, the BLS does not provide OFLC with the wage rates reported to BLS by individual businesses. The identity of respondents and the information that they report to BLS is kept in strict confidence in accordance with BLS Data Integrity Guidelines and with the Confidentiality Information Protection and Statistical Efficiency Act (CIPSEA) of 2002.

In this project, I analyze over 3 million records of H-1B petitions in the period 2011-2016. Next, I dive into the data transformations.

<h2> Dataset Description </h2>

First, I describe the key elements of the data. The data set includes 40 columns in each year's records and the column names completely changed after 2015. My first step was to rename the columns in older records for the relevant columns to match with the newer records. The relevant columns include:

1. **EMPLOYER_NAME**: Name of employer submitting the H1-B application. Used in comparing salaries and number of applications of various employers.

2. **JOB_TITLE**: Title of the job using which we can filter specific job positions for e.g., Data Scientist, Data Engineer etc.

3. **PREVAILING_WAGE**: The prevailing wage for a job position is defined as the average wage paid to similarly employed workers in the requested occupation in the area of intended employment. The prevailing wage is based on the employer’s minimum requirements for the position. (Source). This column will be one of the  key metrics of the data analysis.

4. **WORKSITE_CITY, WORKSITE_STATE**: The foreign worker’s intended area of employment. We will explore the relationship between prevailing wage for Data Scientist position across different locations.

5. **CASE_STATUS**: Status associated with the last significant event or decision. Valid values include “Certified,” “Certified-Withdrawn,” Denied,” and “Withdrawn”. This feature will help us analyze what share of the H-1B visa is taken by different employers/ job positions.

Other important columns include Unit of Pay and whether the Job position is a Full Time position or a Part-Time position.

<h2> Data Transformations </h2>


The main data transformations I performed are as follows:

<h3> Wage Unit of Pay </h3>

While 92% of the records provide Wage at the Year scale, 7.73% provide the information at Hour scale. As only 0.02% of the records have missing information, I remove such records from further analysis. For the remaining records, I convert them to the Year scale.

<h3> Imputing Full-Time position </h3>

Interestingly, 21.6% of the records have missing values regarding the Full Time Position. For filling the missing values, I analyze the relationship of the Prevailing Wage with Full Time Position across the years.

![Full Time Position](/images/h_1b_eda/full_time_position_before_transform.jpeg "Wage Distribution by Position Type (Full-Time/Part-Time)")

<center> Figure 1. Missing values for Full-Time Position </center>

*Observations from Figure 1:*

1. 100% of the records from 2016 have missing values.

2. Expectedly, the median wage for Full time positions are higher than for part-time positions.

Based on the 75% percentile value for Part-Time positions, I select 70000 as the Prevailing Wage cut-off for Full-Time positions with missing values. Accordingly, the missing values are filled.

<h3> Work Site Spelling Corrector </h3>

A significant number of the Worksite values have spelling errors. For example, New York was misspelled New Yrok 16 times, San Francisco misspelled San Fransisco 82 times and Sunnyvale misspelled Suunyvale 11 times. These are just a few examples. In order to correct the spellings in a systematic approach, I implemented a Spell Corrector. This spell corrector uses a probabilistic model which was first implemented in Python by [Peter Norvig][spell-corrector].

To describe briefly, this spell corrector finds out every possible transformation to a given word by 1-edit distance including deleting a letter, interchanging of two adjacent letters, inserting a new letter, replacing a letter with another letter from the English dictionary. This is performed for a single position in a word for a possible candidate correct solution. Once these transformations are obtained, the transformation with the highest occurrence in the list of work sites in our dataset is selected as the correct spelling. The code for this spelling corrector can be found on my [GitHub][my-spell-corrector]. This code uses [hashmap][hashmap] package that maps every worksite with the frequency of occurrence in the dataset.

As Houston is present in Texas, California and few other states, it would be erroneous to consider only the Work site city for this spelling correction. Therefore, I include both worksite city and worksite state to find the frequencies of occurrence before performing the spelling correction.

<h3> Geocoding </h3>

I find out the latitude and longitudes of the work sites. This information will help in creating map plots for the metrics considered in the data analysis.

```r
library(ggmap)

top_sites <- (sites_count$WORKSITE)[1:2500]

site_geocodes <- cbind(geocode(top_sites),top_sites)
```

[ggmap][ggmap] package provides a convenient way of finding out the geocode given a location in string format. However, there is a 2500 request limit per day. Therefore, I find out the geocode only for the top 2500 worksites based on number of H-1B applications observed in our dataset. 96.47% of the records in our dataset are covered by the top 2500 work sites so it was sufficient information for the data analysis.

<h3> Cost of Living Index </h3>

I expect the Wages offered for the same job position might vary significantly based on the cost of living. This will be another component of my data analysis. To collect this data, I used Scrapy package in Python to collect Cost of Living plus Rent index for top cities in the US.

The data was scraped from [Numbeo][numbeo] site comprising of 119 cities. The GitHub code for the Scrapy spider can be found [here][scrapy]. I will discuss web-scraping with Python in a future blog post.

<h2> Conclusion </h2>

After completing all the above transformations, we are ready to begin the data analysis. In the [next blog][h1b-part-2] of this series, I discuss the key insights I drew from the curated data. Thanks for reading!

[nyc_dsa]: https://blog.nycdatascience.com/student-works/h-1b-visa-petitions-exploratory-data-analysis/
[oflc-data]: https://www.foreignlaborcert.doleta.gov/performancedata.cfm
[github]: https://github.com/sharan-naribole/H1B_visa_eda/blob/master/data_processing.Rmd
[spell-corrector]: http://norvig.com/spell-correct.html
[hashmap]: https://cran.r-project.org/web/packages/hashmap/README.html
[my-spell-corrector]: https://github.com/sharan-naribole/H1B_visa_eda/blob/master/spell_correcter.R
[ggmap]: https://github.com/dkahle/ggmap
[numbeo]: https://www.numbeo.com/cost-of-living/country_result.jsp?country=United+States
[scrapy]: https://github.com/sharan-naribole/H1B_visa_eda/blob/master/coli/coli/spiders/coli.py
[h1b-part-I]: https://sharan-naribole.github.io/2017/02/24/h1b-eda-part-I.html
[h1b-part-II]: https://sharan-naribole.github.io/2017/02/26/h1b-eda-part-II.html
[h1b-part-III]: https://sharan-naribole.github.io/2017/02/28/h1b-eda-part-III.html
[h1b-part-IV]: https://sharan-naribole.github.io/2017/03/02/h1b-eda-part-IV.html
