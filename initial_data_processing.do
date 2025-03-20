*-------------------------------------------------------------------------------
*
* Code for the paper:
* "AI brings the finance world a fresh approach to relative valuation"
* by Geertsema, Lu and Stouthuysen (2025)
*
*-------------------------------------------------------------------------------

*-------------------------------------------------------------------------------
* Copyright (C) Paul Geertsema, 2025
* Code for the paper:
* "AI brings a fresh approach to relative valuation"
*
*    This program is free software: you can redistribute it and/or modify
*    it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.

*    This program is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU General Public License for more details.

*    You should have received a copy of the GNU General Public License
*    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*-------------------------------------------------------------------------------

*-------------------------------------------------------------------------------
* file locations
*-------------------------------------------------------------------------------

* Change as needed
global source      C:/Data/DML/Source
global work        C:/Data/HBR/Work

cd $work	

* We retain the most recent 36 months of data
global keep_condition ((mth >= tm(2021m1)) & (mth <= tm(2023m12)))
	
* -----------------------------------------------------------------------------
* CRSP monthly
* -----------------------------------------------------------------------------

* CRSP monthly file (from WRDS)
use ${source}/crsp_monthly, clear

* change variables to lowercase
foreach v of varlist * {
	local newv = lower("`v'")
	qui di "`v' `newv'"
	qui ren `v' `newv'
}

* standard finance filters:

* keep common domestic US stocks
keep if shrcd == 10 | shrcd == 11 | shrcd == 12

* traded on NYZE, Nasdaq and Amex
keep if exchcd == 1 | exchcd == 2 | exchcd == 3

* create a month variables
gen mth = mofd(date)
format mth %tm
la var mth "Return month"

* keep relevant sample 
keep if $keep_condition

save crsp_temp, replace

* create a market cap variable.
* note: shrout is in thousands
* note: marketcap is in millions USD, to match compustat
gen marketcap = abs(prc) * (shrout*1000) / 1000000
la var marketcap "Market cap in millions"

* keep the largest permno by permco as the "representative" security

* sort data by month, permco, and marketcap (descending)
gsort mth permco -marketcap

* for each month-permco combination, flag the observation with largest marketcap
by mth permco: gen keep_flag = (_n == 1)

* Set permno to missing for all observations except the largest in each permco group
replace permno = . if keep_flag == 0
drop keep_flag

* check that we have clean panel by permco and mth
duplicates report permco mth if !missing(permno)

* gcollapse is part of of gtools
* https://gtools.readthedocs.io/en/latest/
gcollapse (sum) marketcap, by(permno mth) 

* merge in the ticker for the stock
merge 1:1 permno mth using crsp_temp, nogen keep(master match) keepusing(ticker)

* save as temporary file for later use
save crsp_monthly, replace

* -----------------------------------------------------------------------------
* Chen and Zimmerman data
* -----------------------------------------------------------------------------

* we are using the open source data from Chen and Zimmerman (2021)
* "Open Source Cross-Sectional Asset Pricing" in Critical Finance Review
* https://www.nowpublishers.com/article/Details/CFR-0112
* for the data see:
* https://www.openassetpricing.com/

use ${source}/cz, clear

* create mth variable
ren mdate mth
format mth %tm

* keep only the relevant sample
keep if $keep_condition

* the valuation target is the logarithm of the book to market ratio (as of December)
* we take logs to mitigate the effect of outliers
* this approach has been validated in Geertsema & Lu (2023), "Relative valuation with machine learning"
* see https://onlinelibrary.wiley.com/doi/full/10.1111/1475-679X.12464

gen target = ln(bmdec) if bmdec > 0

* only keep observations for which we have a valid target
drop if missing(target)

* these are the variables we use from Chen & Zimmerman (2022)
* note: we exclude indicator (binary) variables
local preds abnormalaccruals accruals activism1 analystrevision assetgrowth bookleverage cash cboperprof changeinrecommendation chassetturnover cheq chinv chinvia chnncoa chnwc chtax compositedebtissuance delbreadth delcoa delcol delequ delfinl dellti delnetfin dnoa earningsconsistency earningsforecastdisparity earningsstreak earningssurprise earnsupbig exclexp feps fgr5yrlag forecastdispersion gp gradexp grcapx grcapx3y investment investppeinv invgrowth meanrankrevgrowth netdebtfinance netequityfinance noa opleverage orderbacklog orderbacklogchg orgcap pctacc pcttotacc ps rdability rdcap rev6 revenuesurprise roaq sfe shareiss1y shareiss5y tang tax totalaccruals xfin activism2 brandinvest deldrc grltnoa grsaletogrinv grsaletogroverhead io_shortinterest numearnincrease operprof operprofrd predictedfe realestate roe varcf

* merge in the CRSP data 
* we need the marketcap for filtering the largest companies
* and the ticker for easy identification
* if you don't need these, then there is no need for CRSP data

merge 1:1 permno mth using crsp_monthly, nogen keep(match) keepusing(marketcap ticker)

* keep the relevant variables
keep permno ticker mth target marketcap `preds'

* only keep observations for which marketcap is not missing
keep if !missing(marketcap)

* -----------------------------------------------------------------------------
* Largest companies 
* -----------------------------------------------------------------------------

* the idea here is to only keep the very largest firms
* since our code is for exposition, we need to keep the number of 
* firms manageble

* we'll start with 100, but after retaining valid observations only
* it will be 73. (We only keep firms with complete time-series, see below)

local number_of_firms 100

* in each month sort by market cap descending
gsort mth -marketcap
by mth: gen rank = _n
* and only keep the largest `number of firms' in that month
gen keeper = 1 if rank <= `number_of_firms'
keep if keeper == 1

* get unique number of firms (should be 36 and it is)
levelsof mth, local(months)

* get the unique list of tickers in the data
levelsof ticker, local(tickers)

local m = wordcount("`months'")

* for each ticker, drop the ticker if it does not have the full set of 
* months in the data. this guarantees a properly complete panel

foreach t in `tickers' {
	qui su if ticker == "`t'"
	local cnt = r(N)
	if `cnt' != `m' {
		drop if ticker == "`t'"
		di "dropped `t'"
	}
}


table ticker

* save as a temporary file
save cz_temp, replace

* -----------------------------------------------------------------------------
* Target and features
* -----------------------------------------------------------------------------

* load temporary data file
use cz_temp, clear

* set up and enforce panel data structure
tsset permno mth

* only keep variables that are less than 50% missing
* the ML approach we use (LightGBM) can handle missing
* data internally, but with more than 50% missing, there is not much
* marginal value in the data

qui su permno
local total_obs = r(N)

* only keep if 50%+ non-missing
foreach v in `preds' {
	qui su `v', meanonly
	if r(N) < `total_obs' * 0.50 {
		drop `v'
		di "dropped `v'"
	}
}

* get the list of final predictors
unab preds : abnormalaccruals - roaq

drop keeper rank

* put in order
order permno ticker mth target marketcap `preds'

* save final data set to be used by ML code (in python, a seperate file)
save combined, replace
