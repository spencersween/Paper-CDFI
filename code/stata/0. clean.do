********************************************************************************
***                                                                          ***
***    CDFI LENDING IMPACT ON ENTREPRENEURSHIP: DATA CONSTRUCTION PIPELINE  ***
***                                                                          ***
***    Author:  Spencer Sween                                               ***
***    Updated: January 2026                                                ***
***                                                                          ***
***    Purpose: This do-file constructs a ZIP-code level panel dataset      ***
***             linking CDFI lending activity to entrepreneurship outcomes  ***
***             using data from multiple sources (1988-2016).               ***
***                                                                          ***
********************************************************************************

clear all
set more off
cls


********************************************************************************
***                                                                          ***
***          SECTION 0: PROJECT SETUP (UNCOMMENT FOR FIRST RUN)            ***
***                                                                          ***
********************************************************************************

/* --------------------------------------------------------------------------
   0.1 Create Clean Project Directory Structure
   
   Description: Sets up organized folder structure for new project
   Instructions: 
     1. Change NEW_PROJECT_PATH to your desired location
     2. Uncomment this entire section (remove leading /* and trailing */)
     3. Run once to create directories and copy files
     4. Re-comment this section after successful setup
-------------------------------------------------------------------------- */

/*

* -------- Define paths --------
global OLD_PROJECT "/Users/spencersween/Dropbox/Paper -- CDFI -- Sween 2026"
global NEW_PROJECT "/Users/spencersween/Dropbox/Paper -- CDFI -- Sween 2026"

* -------- Create directory structure --------
capture mkdir "${NEW_PROJECT}"
capture mkdir "${NEW_PROJECT}/code"
capture mkdir "${NEW_PROJECT}/data"
capture mkdir "${NEW_PROJECT}/data/raw"
capture mkdir "${NEW_PROJECT}/data/raw/cdfi_transactions"
capture mkdir "${NEW_PROJECT}/data/raw/entrepreneurship"
capture mkdir "${NEW_PROJECT}/data/raw/crosswalks"
capture mkdir "${NEW_PROJECT}/data/raw/covariates"
capture mkdir "${NEW_PROJECT}/data/raw/labormarket"
capture mkdir "${NEW_PROJECT}/data/raw/downloads"
capture mkdir "${NEW_PROJECT}/data/intermediate"
capture mkdir "${NEW_PROJECT}/data/analysis"
capture mkdir "${NEW_PROJECT}/outputs"
capture mkdir "${NEW_PROJECT}/outputs/figures"
capture mkdir "${NEW_PROJECT}/outputs/tables"

display as text _n "Directory structure created successfully!"


* -------- Copy CDFI Transaction Data --------
display as text _n "Copying CDFI transaction data..."

* FY2003-2015 files (5 parts)
forvalues i = 1(1)5 {
    shell cp "${OLD_PROJECT}/Data/CDFI_Transactions/FY2015_Data_Documentation_Instruction/releaseTLR_fy03_15(`i'of5).csv" ///
             "${NEW_PROJECT}/data/raw/cdfi_transactions/"
}

* FY2018 file
shell cp "${OLD_PROJECT}/Data/CDFI_Transactions/FY2018_Data_Documentation_Instruction/releaseTLR_fy18.csv" ///
         "${NEW_PROJECT}/data/raw/cdfi_transactions/"

* FY2019 file
shell cp "${OLD_PROJECT}/Data/CDFI_Transactions/FY2019_Data_Documentation_Instruction/releaseTLR_fy19.csv" ///
         "${NEW_PROJECT}/data/raw/cdfi_transactions/"

* FY2020 file
shell cp "${OLD_PROJECT}/Data/CDFI_Transactions/FY2020_Data_Documentation_Instruction/releaseTLR_fy20.csv" ///
         "${NEW_PROJECT}/data/raw/cdfi_transactions/"

* FY2021 file
shell cp "${OLD_PROJECT}/Data/CDFI_Transactions/FY2021_Data_Documentation_Instruction/releaseTLR_fy21.csv" ///
         "${NEW_PROJECT}/data/raw/cdfi_transactions/"

display as text "CDFI data copied."


* -------- Copy Entrepreneurship Data --------
display as text _n "Copying entrepreneurship data..."

shell cp "${OLD_PROJECT}/Data/Entrepreneurship_SCP/Entrepreneurship_by_ZIP_Code_academic.dta" ///
         "${NEW_PROJECT}/data/raw/entrepreneurship/"

display as text "Entrepreneurship data copied."


* -------- Copy Geographic Crosswalks --------
display as text _n "Copying geographic crosswalks..."

* Census tract 2000-2010 crosswalk
shell cp "${OLD_PROJECT}/Data/Crosswalks/nhgis_tr2000_tr2010/nhgis_tr2000_tr2010.csv" ///
         "${NEW_PROJECT}/data/raw/crosswalks/"

* Tract-ZIP crosswalk (from Downloads folder)
shell cp "/Users/spencersween/Downloads/tract_zip_032013.xlsx" ///
         "${NEW_PROJECT}/data/raw/crosswalks/"

* ZIP-County crosswalk (from Downloads folder)
shell cp "/Users/spencersween/Downloads/zip_county_032013.xlsx" ///
         "${NEW_PROJECT}/data/raw/crosswalks/"

* ZIP-ZCTA crosswalk (from Downloads folder)
shell cp "/Users/spencersween/Downloads/ZIP Code to ZCTA Crosswalk.xlsx" ///
         "${NEW_PROJECT}/data/raw/crosswalks/"

display as text "Crosswalks copied."


* -------- Copy Covariate Data --------
display as text _n "Copying covariate data..."

* ACS Demographics (ICPSR)
shell cp "${OLD_PROJECT}/Data/Covariates_ICPSR38528/DS0003/38528-0003-Data.dta" ///
         "${NEW_PROJECT}/data/raw/covariates/"

* Bank structure data
shell cp "${OLD_PROJECT}/Data/Stata Cleaning/bank_structure.dta" ///
         "${NEW_PROJECT}/data/raw/covariates/"

* Land cover data
shell cp "${OLD_PROJECT}/Data/Stata Cleaning/landcover.dta" ///
         "${NEW_PROJECT}/data/raw/covariates/"

display as text "Covariate data copied."


* -------- Copy Labor Market Data --------
display as text _n "Copying labor market data..."

* County Business Patterns (ZIP level)
shell cp "/Users/spencersween/Downloads/zbp_1994_2016_by_zip.csv" ///
         "${NEW_PROJECT}/data/raw/labormarket/"

display as text "Labor market data copied."


* -------- Copy This Do-File to Code Folder --------
display as text _n "Copying master do-file..."

shell cp "${OLD_PROJECT}/[CURRENT_DO_FILE_NAME].do" ///
         "${NEW_PROJECT}/code/cdfi_analysis_master.do"

display as text "Do-file copied."


* -------- Create README file --------
file open readme using "${NEW_PROJECT}/README.txt", write replace
file write readme "CDFI LENDING IMPACT ON ENTREPRENEURSHIP" _n
file write readme "=======================================" _n _n
file write readme "Project created: `c(current_date)'" _n _n
file write readme "DIRECTORY STRUCTURE:" _n
file write readme "-------------------" _n
file write readme "code/                  - All Stata do-files" _n
file write readme "data/raw/              - Original source data (DO NOT MODIFY)" _n
file write readme "data/intermediate/     - Cleaned/processed datasets" _n
file write readme "data/analysis/         - Final analysis datasets" _n
file write readme "outputs/figures/       - All graphs and figures" _n
file write readme "outputs/tables/        - All tables and results" _n _n
file write readme "GETTING STARTED:" _n
file write readme "---------------" _n
file write readme "1. Open code/cdfi_analysis_master.do" _n
file write readme "2. Update working directory path (line ~50)" _n
file write readme "3. Run the do-file to build analysis dataset" _n _n
file write readme "DATA SOURCES:" _n
file write readme "------------" _n
file write readme "- CDFI Fund Transaction Level Reports (FY2003-2021)" _n
file write readme "- Startup Cartography Project (Entrepreneurship data)" _n
file write readme "- NHGIS Census Tract Crosswalks" _n
file write readme "- HUD-USPS ZIP-Tract/County Crosswalks" _n
file write readme "- ACS 2008-2012 Demographics (ICPSR 38528)" _n
file write readme "- County Business Patterns (1994-2016)" _n
file write readme "- Bank Structure Data (NaNDA)" _n
file write readme "- Land Cover Data" _n
file close readme

display as text _n _n "=========================================="
display as text "PROJECT SETUP COMPLETE!"
display as text "=========================================="
display as text _n "New project location: ${NEW_PROJECT}"
display as text _n "Next steps:"
display as text "  1. Navigate to: ${NEW_PROJECT}"
display as text "  2. Open: code/cdfi_analysis_master.do"
display as text "  3. Update working directory path"
display as text "  4. Re-comment this setup section"
display as text "  5. Run analysis pipeline"
display as text _n "===========================================" _n

*/

* End of commented-out setup section


* -------- Set working directory --------

cd "/Users/spencersween/Dropbox/Paper -- CDFI -- Sween 2026"

* Set global paths for entire project
global code       "code"
global rawdata    "data/raw"
global intdata    "data/intermediate"
global analysis   "data/analysis"
global outputs    "outputs"
global figures    "outputs/figures"
global tables     "outputs/tables"


********************************************************************************
***                                                                          ***
***                     SECTION 1: RAW DATA IMPORT                          ***
***                                                                          ***
********************************************************************************

/* --------------------------------------------------------------------------
   1.1 CDFI Transaction-Level Records (TLR) Data
   
   Description: Imports and combines CDFI lending records from FY2003-2021
   Source: CDFI Fund Transaction Level Reports
   Note: Currently commented out - uncomment for fresh data import
-------------------------------------------------------------------------- */

/*
clear all
gen x = .
save "${intdata}/raw_cdfi_data.dta", replace

* Import FY2003-2015 data (5 files)
forvalues i = 1(1)5 {
    import delimited "${rawdata}/cdfi_transactions/releaseTLR_fy03_15(`i'of5).csv", ///
        clear varn(1)
    append using "${intdata}/raw_cdfi_data.dta", force
    save "${intdata}/raw_cdfi_data.dta", replace
}

* Import FY2018 data
import delimited "${rawdata}/cdfi_transactions/releaseTLR_fy18.csv", ///
    clear varn(1)
append using "${intdata}/raw_cdfi_data.dta", force
save "${intdata}/raw_cdfi_data.dta", replace

* Import FY2019 data
import delimited "${rawdata}/cdfi_transactions/releaseTLR_fy19.csv", ///
    clear varn(1)
append using "${intdata}/raw_cdfi_data.dta", force
save "${intdata}/raw_cdfi_data.dta", replace

* Import FY2020 data
import delimited "${rawdata}/cdfi_transactions/releaseTLR_fy20.csv", ///
    clear varn(1)
append using "${intdata}/raw_cdfi_data.dta", force
save "${intdata}/raw_cdfi_data.dta", replace

* Import FY2021 data
import delimited "${rawdata}/cdfi_transactions/releaseTLR_fy21.csv", ///
    clear varn(1)
append using "${intdata}/raw_cdfi_data.dta", force
drop x
save "${intdata}/raw_cdfi_data.dta", replace
*/


/* --------------------------------------------------------------------------
   1.2 Startup Cartography Project (SCP) ZIP Codes
   
   Description: Extracts valid ZIP codes from SCP entrepreneurship data
   Source: Entrepreneurship by ZIP Code (Academic Dataset)
-------------------------------------------------------------------------- */

use "${rawdata}/entrepreneurship/Entrepreneurship_by_ZIP_Code_academic.dta", clear
destring zipcode, gen(zip)
keep zip
gduplicates drop
save "${intdata}/scp_zips.dta", replace


********************************************************************************
***                                                                          ***
***                 SECTION 2: GEOGRAPHIC CROSSWALKS                        ***
***                                                                          ***
********************************************************************************

/* --------------------------------------------------------------------------
   2.1 Census Tract 2000 to 2010 Crosswalk
   
   Description: Maps 2000 census tracts to 2010 census tracts with weights
   Source: NHGIS (National Historical GIS)
-------------------------------------------------------------------------- */

import delimited "${rawdata}/crosswalks/nhgis_tr2000_tr2010.csv", clear

* Format FIPS codes as 11-character strings
tostring tr2000ge, gen(fips2000) format(%20.0f)
tostring tr2010ge, gen(fips2010) format(%20.0f)
replace fips2000 = "0" + fips2000 if length(fips2000) == 10
replace fips2010 = "0" + fips2010 if length(fips2010) == 10

* Keep relevant variables
hashsort fips2000 parea
keep fips2000 fips2010 parea
rename fips2010 fips2010_cross
rename parea w_2000_to_2010

save "${intdata}/tract_2000_to_2010.dta", replace


/* --------------------------------------------------------------------------
   2.2 Census Tract to ZIP Code Crosswalk
   
   Description: Maps census tracts to ZIP codes with allocation ratios
   Source: HUD-USPS ZIP-Tract Crosswalk (Q1 2013)
   Note: Restricted to SCP ZIP codes only
-------------------------------------------------------------------------- */

import excel "${rawdata}/crosswalks/tract_zip_032013.xlsx", ///
    clear firstrow
rename *, lower
destring zip, replace

* Restrict to SCP ZIPs
fmerge m:1 zip using "${intdata}/scp_zips.dta", keep(3) nogen

* Keep relevant variables
hashsort tract zip tot_ratio
keep tract zip tot_ratio
rename tot_ratio w_tract_to_zip

save "${intdata}/tract_to_zip.dta", replace


/* --------------------------------------------------------------------------
   2.3 ZIP Code to County Crosswalk
   
   Description: Maps ZIP codes to counties with allocation ratios
   Source: HUD-USPS ZIP-County Crosswalk (Q1 2013)
   Creates both many-to-many and one-to-one mappings
-------------------------------------------------------------------------- */

import excel "${rawdata}/crosswalks/zip_county_032013.xlsx", ///
    clear firstrow
rename *, lower
destring zip, replace

* Restrict to SCP ZIPs
fmerge m:1 zip using "${intdata}/scp_zips.dta", keep(3) nogen

* Save full crosswalk (many-to-many)
hashsort zip county tot_ratio
keep zip county tot_ratio
rename tot_ratio w_zip_to_county
save "${intdata}/zip_to_county.dta", replace

* Create one-to-one mapping (keep largest county share per ZIP)
hashsort zip w_zip_to_county
by zip: keep if _n == _N
save "${intdata}/zip_to_one_county.dta", replace


/* --------------------------------------------------------------------------
   2.4 ZIP Code to ZCTA Crosswalk
   
   Description: Maps ZIP codes to ZIP Code Tabulation Areas (ZCTAs)
   Note: ZCTAs are census geographic units that approximate ZIP codes
-------------------------------------------------------------------------- */

import excel "${rawdata}/crosswalks/ZIP Code to ZCTA Crosswalk.xlsx", ///
    clear firstrow
rename *, lower
rename zip_code zip
keep zip zcta
gduplicates drop
destring *, replace

* Restrict to SCP ZIPs
fmerge m:1 zip using "${intdata}/scp_zips.dta", keep(3) nogen

save "${intdata}/zip_to_zcta.dta", replace


********************************************************************************
***                                                                          ***
***             SECTION 3: CDFI TREATMENT VARIABLE CONSTRUCTION             ***
***                                                                          ***
********************************************************************************

/* --------------------------------------------------------------------------
   3.1 County-Level CDFI Treatment Timing
   
   Description: Identifies the first year of CDFI activity in each county
   Process:
     - Cleans CDFI transaction dates
     - Restricts to business loans (1996-2014)
     - Maps to 2010 census tracts using crosswalk
     - Collapses to county-year level
     - Identifies first treatment year (G) per county
-------------------------------------------------------------------------- */

use "${intdata}/raw_cdfi_data.dta", clear

* -------- Clean and standardize dates --------
split dateclosed, parse("-")
gen year = .
replace year = 2000 + real(dateclosed3) if inrange(real(dateclosed3), 0, 21)
replace year = 1900 + real(dateclosed3) if inrange(real(dateclosed3), 96, 99)
drop if missing(year)
keep if inrange(year, 1996, 2014)

* -------- Standardize census tract identifiers --------
tostring projectfipscode_2010, gen(fips2010) format(%20.0f)
replace fips2010 = "0" + fips2010 if length(fips2010) == 10
replace fips2010 = "" if fips2010 == "."

gen fips2000 = projectfipscode_2000
replace fips2000 = "0" + fips2000 if length(fips2000) == 10

* -------- Drop observations with missing locations --------
drop if fips2000 == "NONE" & missing(fips2010)
drop if missing(fips2000) & missing(fips2010)

* -------- Restrict to business loans --------
keep if inlist(purpose, "BUSFIXED", "BUSINESS", "BUSWORKCAP", "MICRO", ///
                        "OTHER", "RECOCOM", "RERHCOM")

* -------- Generate aggregate lending volume graph --------
preserve
    keep originalamount year
    replace originalamount = originalamount / 1000000000
    gcollapse (sum) originalamount, by(year)
    
    tsset year
    set scheme gg_viridis
    su originalamount
    
    tsline originalamount, ///
        yline(`r(mean)', lw(0.5) lp(dash) lc(red)) ///
        lc(black) lw(1) ///
        xtitle("Year") ///
        ytitle("Annual Lending Volumes (Billions $)") ///
        title("TLR-Reported CDFI Business Lending, 1996-2014") ///
        xlabel(#10, labsize(small)) ///
        text(1 2000 "Sample Period Average", size(small) color(red)) ///
        ylabel(0(1)5, labsize(small))
    
    graph export "${figures}/tlr_cdfi_lending.png", replace
restore

* -------- Prepare for tract aggregation --------
keep fips2000 fips2010 year

* -------- Update 2000 FIPS codes to 2010 standard --------
joinby fips2000 using "${intdata}/tract_2000_to_2010.dta", unmatched(master)
replace fips2010 = fips2010_cross if missing(fips2010)
drop if missing(fips2010)
replace w_2000_to_2010 = 1 if missing(w_2000_to_2010)

* -------- Collapse to 2010 census tracts --------
gen num_cdfi = 1 * w_2000_to_2010
gcollapse (sum) num_cdfi, by(fips2010 year) labelformat(" ")
drop if num_cdfi == 0

* -------- Aggregate to county-year level --------
gen county = substr(fips2010, 1, 5)
gcollapse (sum) num_cdfi, by(county year) labelformat(" ")
replace num_cdfi = ceil(num_cdfi)

* -------- Create balanced panel (all counties, all years) --------
hashsort county year
keep if inrange(year, 1996, 2014)
gegen id = group(county)
xtset id year
tsfill, full

* Fill in county identifiers
hashsort id year
by id: carryforward county, replace
hashsort id -year
by id: carryforward county, replace
drop id

* Fill in missing CDFI counts with zeros
hashsort county year
replace num_cdfi = 0 if missing(num_cdfi)

* -------- Identify first treatment year per county --------
gen t = year if num_cdfi > 0
by county: gegen G = min(t)
gcollapse (min) G, by(county)

* -------- Save county treatment data --------
save "${intdata}/county_cdfi_treatment_data.dta", replace


/* --------------------------------------------------------------------------
   3.2 ZIP-Level CDFI Intensity
   
   Description: Counts CDFI loans per ZIP-year using tract-to-ZIP crosswalk
   Process: Similar to 3.1 but maps to ZIP codes instead of counties
-------------------------------------------------------------------------- */

use "${intdata}/raw_cdfi_data.dta", clear

* -------- Replicate date and tract cleaning from Section 3.1 --------
split dateclosed, parse("-")
gen year = .
replace year = 2000 + real(dateclosed3) if inrange(real(dateclosed3), 0, 21)
replace year = 1900 + real(dateclosed3) if inrange(real(dateclosed3), 96, 99)
drop if missing(year)
keep if inrange(year, 1996, 2014)

tostring projectfipscode_2010, gen(fips2010) format(%20.0f)
replace fips2010 = "0" + fips2010 if length(fips2010) == 10
replace fips2010 = "" if fips2010 == "."

gen fips2000 = projectfipscode_2000
replace fips2000 = "0" + fips2000 if length(fips2000) == 10

drop if fips2000 == "NONE" & missing(fips2010)
drop if missing(fips2000) & missing(fips2010)

keep if inlist(purpose, "BUSFIXED", "BUSINESS", "BUSWORKCAP", "MICRO", ///
                        "OTHER", "RECOCOM", "RERHCOM")

keep fips2000 fips2010 year

* -------- Apply 2000-to-2010 tract crosswalk --------
joinby fips2000 using "${intdata}/tract_2000_to_2010.dta", unmatched(master)
replace fips2010 = fips2010_cross if missing(fips2010)
drop if missing(fips2010)
replace w_2000_to_2010 = 1 if missing(w_2000_to_2010)

* -------- Collapse to 2010 tracts --------
gen num_cdfi = 1 * w_2000_to_2010
gcollapse (sum) num_cdfi, by(fips2010 year) labelformat(" ")
drop if num_cdfi == 0

* -------- Map tracts to ZIP codes --------
rename fips2010 tract
joinby tract using "${intdata}/tract_to_zip.dta", unmatched(none)

* -------- Collapse to ZIP-year level --------
replace num_cdfi = num_cdfi * w_tract_to_zip
gcollapse (sum) num_cdfi, by(zip year) labelformat(" ")
replace num_cdfi = ceil(num_cdfi)
drop if num_cdfi == 0

* -------- Create balanced ZIP-year panel --------
hashsort zip year
keep if inrange(year, 1996, 2014)
gegen id = group(zip)
xtset id year
tsfill, full

hashsort id year
by id: carryforward zip, replace
hashsort id -year
by id: carryforward zip, replace
drop id

hashsort zip year
replace num_cdfi = 0 if missing(num_cdfi)

* -------- Merge to any county (for clustering purposes) --------
joinby zip using "${intdata}/zip_to_county.dta", unmatched(none)
hashsort zip year w_zip_to_county
by zip year: keep if _n == _N
drop w_zip_to_county

* -------- Save ZIP intensity data --------
save "${intdata}/zip_cdfi_intensity_data.dta", replace


/* --------------------------------------------------------------------------
   3.3 ZIP-Level Treatment: County Emergence
   
   Description: Assigns county-level treatment timing to ZIP codes
   Method: Uses primary county (largest geographic overlap)
-------------------------------------------------------------------------- */

use "${intdata}/scp_zips.dta", clear

* Merge to county crosswalk and keep largest share
merge 1:m zip using "${intdata}/zip_to_county.dta", keep(3) nogen
hashsort zip w_zip_to_county
by zip: keep if _n == _N
drop w_zip_to_county

* Bring in county-level treatment timing
merge m:1 county using "${intdata}/county_cdfi_treatment_data.dta", ///
    keep(1 3) nogen
gcollapse (min) G = G, by(zip)
replace G = 0 if missing(G)
rename G G_county

save "${intdata}/zip_cdfi_treatment_county.dta", replace


/* --------------------------------------------------------------------------
   3.4 ZIP-Level Treatment: Final Assignment
   
   Description: Combines county emergence and ZIP intensity measures
   Logic:
     - G_county = first CDFI activity in ZIP's primary county
     - G_intensity = first CDFI activity observed in the ZIP itself
     - G_final = earlier of G_county and G_intensity (if both positive)
-------------------------------------------------------------------------- */

use "${intdata}/scp_zips.dta", clear

* Identify primary county per ZIP
merge 1:m zip using "${intdata}/zip_to_county.dta", keep(3) nogen
hashsort zip w_zip_to_county
by zip: keep if _n == _N
drop w_zip_to_county

* Merge ZIP-level intensity data
merge 1:m zip using "${intdata}/zip_cdfi_intensity_data.dta", keep(1 3) nogen
replace year = 1988 if missing(year)
xtset zip year
tsfill, full

* Forward/backward fill county identifiers
hashsort zip year
by zip: carryforward county, replace
hashsort zip -year
by zip: carryforward county, replace
hashsort zip year
replace num_cdfi = 0 if missing(num_cdfi)

* Identify first observed CDFI activity per ZIP
hashsort zip year
by zip: gegen G_intensity = min(cond(num_cdfi > 0, year, .))
replace G_intensity = 0 if missing(G_intensity)

* Merge county-level treatment timing
merge m:1 zip using "${intdata}/zip_cdfi_treatment_county.dta", keep(3) nogen

* Compute final treatment year
gen G_final = 0
replace G_final = G_county if G_county > 0
replace G_final = G_intensity if G_county > 0 & G_intensity > 0 & ///
                                  G_county > G_intensity
replace G_final = G_intensity if G_county == 0 & G_intensity > 0

* Save final treatment assignment
keep zip year G_intensity G_county G_final num_cdfi
order zip year G_intensity G_county G_final num_cdfi
hashsort zip year

save "${intdata}/zip_cdfi_treatment_data_final.dta", replace


********************************************************************************
***                                                                          ***
***                   SECTION 4: CONTROL VARIABLES                          ***
***                                                                          ***
********************************************************************************

/* --------------------------------------------------------------------------
   4.1 ACS 2008-2012 Demographics
   
   Description: Demographic and socioeconomic covariates at ZCTA level
   Source: ICPSR 38528 (ACS 5-year estimates)
-------------------------------------------------------------------------- */

use "${rawdata}/covariates/38528-0003-Data.dta", clear

* Keep only 2008-2012 variables
keep ZCTA10 *08_12
rename *, lower
rename zcta10 zcta

* Recode missing values to zero and add prefix
foreach v of varlist totpop08_12-ethnicimmigrant08_12 {
    replace `v' = 0 if missing(`v')
    rename `v' X_`v'
    format X_`v' %11.0g
}

* Merge ZCTA to ZIP
destring zcta, replace
hashsort zcta
merge 1:m zcta using "${intdata}/zip_to_zcta.dta", keep(3) nogen

order zip zcta X*
hashsort zip
save "${intdata}/zip_sfr_covariates.dta", replace


/* --------------------------------------------------------------------------
   4.2 NaNDA Population, Banks, and Land Cover
   
   Description: Time-varying population, banking presence, and land use
   Sources:
     - Bank structure data (NaNDA)
     - Land cover data
     - ACS demographics (merged above)
-------------------------------------------------------------------------- */

use "${rawdata}/covariates/bank_structure.dta", clear

* Recode missing population and land area
replace totpop = 0 if missing(totpop)
replace aland10 = 0 if missing(aland10)

* Shift years for panel alignment
hashsort zcta year
replace year = 1988 if year == 2017
replace year = 1989 if year == 2018
replace totpop = . if year == 1988
replace aland10 = . if year == 1988
replace totpop = . if year == 1989
replace aland10 = . if year == 1989

* Restrict to sample period
keep if inrange(year, 1988, 2014)

* Merge land cover data
fmerge 1:1 zcta year using "${rawdata}/covariates/landcover.dta", keep(3) nogen

* Forward/backward fill time-invariant variables
hashsort zcta -year
by zcta: carryforward totpop aland10 prop_*, replace
hashsort zcta year
by zcta: carryforward totpop aland10 prop_*, replace

* Map ZCTA to ZIP and merge demographics
joinby zcta using "${intdata}/zip_to_zcta.dta"
fmerge m:1 zip zcta using "${intdata}/zip_sfr_covariates.dta", ///
    keep(3) nogen

order zip zcta year, first
save "${intdata}/zip_sfr_covariates.dta", replace


/* --------------------------------------------------------------------------
   4.3 County Business Patterns (Labor Market Data)
   
   Description: Establishments, employment, and wages at ZIP level
   Source: ZIP Business Patterns 1994-2016
-------------------------------------------------------------------------- */

import delimited "${rawdata}/labormarket/zbp_1994_2016_by_zip.csv", ///
    clear varn(1)
rename zipcode zip

* Restrict to SCP ZIPs
fmerge m:1 zip using "${intdata}/scp_zips.dta", keep(3) nogen

* Remove duplicates
hashsort zip year
by zip year: keep if _n == _N

* Create balanced panel
xtset zip year
tsfill, full

* Recode zeros to missing for interpolation
foreach v of varlist estab emp payann payqtr1 {
    replace `v' = . if `v' == 0
}

* Forward/backward fill values
hashsort zip -year
by zip: carryforward *, replace
hashsort zip year
by zip: carryforward *, replace

* Recode missing back to zero
foreach v of varlist estab emp payann payqtr1 {
    replace `v' = 0 if missing(`v')
}

* Impute employment if missing but payroll exists
replace emp = estab if emp == 0 & payann > 0

* Compute average wage
gen wage = floor((payann * 1000) / emp)
replace wage = 0 if missing(wage)

keep zip year estab emp wage
save "${intdata}/zip_labormarket.dta", replace


********************************************************************************
***                                                                          ***
***              SECTION 5: FINAL ANALYSIS DATASET CONSTRUCTION             ***
***                                                                          ***
********************************************************************************

/* --------------------------------------------------------------------------
   5.1 Load Entrepreneurship Data
   
   Description: ZIP-level entrepreneurship outcomes from Startup Cartography
   Variables:
     - sfr: Startup Formation Rate (new business registrations)
     - growth: Business growth/expansions
     - eqi: Entrepreneurial Quality Index
-------------------------------------------------------------------------- */

use "${rawdata}/entrepreneurship/Entrepreneurship_by_ZIP_Code_academic.dta", clear

rename *, lower
destring zipcode, gen(zip)

* Round and weight variables
replace sfr = round(sfr)
replace growth = round(growth)
replace eqi = eqi * sfr

* Collapse to ZIP-year level
gcollapse (sum) sfr growth eqi, by(zip year)

* Recompute average EQI
replace sfr = round(sfr)
replace growth = round(growth)
replace eqi = eqi / sfr
replace eqi = 0 if missing(eqi)


/* --------------------------------------------------------------------------
   5.2 Create Balanced Panel
-------------------------------------------------------------------------- */

hashsort zip year
xtset zip year
tsfill, full

* Fill missing values with zeros
foreach v in growth sfr eqi {
    replace `v' = 0 if missing(`v')
}


/* --------------------------------------------------------------------------
   5.3 Merge Treatment Variables
-------------------------------------------------------------------------- */

fmerge 1:1 zip year using "${intdata}/zip_cdfi_treatment_data_final.dta", ///
    keep(3) nogen


/* --------------------------------------------------------------------------
   5.4 Merge Covariates and Geographic Identifiers
-------------------------------------------------------------------------- */

* Demographics, population, land cover
fmerge 1:1 zip year using "${intdata}/zip_sfr_covariates.dta", keep(3) nogen

* County and state identifiers
fmerge m:1 zip using "${intdata}/zip_to_one_county.dta", ///
    keep(3) nogen keepusing(county)
gen state = substr(county, 1, 2)

* Labor market data
fmerge 1:1 zip year using "${intdata}/zip_labormarket.dta", keep(1 3)
hashsort zip
by zip: gegen max_merge = max(_merge == 3)
keep if max_merge == 1
drop _merge max_merge

* Drop ZIPs with missing wage data
by zip: gegen zero_wage = max(wage == 0)
drop if zero_wage
drop zero_wage


/* --------------------------------------------------------------------------
   5.5 Clean Population Data and Remove Low-Population Areas
-------------------------------------------------------------------------- */

hashsort zcta year

* Recode zero population to missing for interpolation
replace totpop = . if totpop == 0
replace X_totpop08_12 = . if X_totpop08_12 == 0
replace totpop = round(totpop)
replace X_totpop08_12 = round(X_totpop08_12)

* Forward fill population
by zcta: carryforward totpop X_totpop08_12, replace

* Recode missing back to zero
foreach v in totpop X_totpop08_12 {
    replace `v' = 0 if missing(`v')
}

* Drop low-population ZCTAs
hashsort zcta year
by zcta: gegen low_pop = max(totpop <= 100 | X_totpop08_12 <= 100)
drop if low_pop == 1
drop low_pop


/* --------------------------------------------------------------------------
   5.6 Create Panel Identifiers and Treatment Indicators
-------------------------------------------------------------------------- */

gen id = zip
gen time = year
gen group = G_final

* Clustering variables
gen cluster_zcta = zip
gen cluster_county = county
gen cluster_state = state
destring cluster_*, replace

* Treatment indicators
gen i_treat = (G_final > 0)
gen i_treat_post = (year >= G_final & G_final > 0)


/* --------------------------------------------------------------------------
   5.7 Generate Outcome Variables
-------------------------------------------------------------------------- */

* Count outcomes
gen y_sfr = sfr
gen y_eqi = eqi
gen y_growth = growth
gen y_cdfi = num_cdfi

* Per-capita outcomes
gen y_sfr_pc = (sfr / totpop) * 1000
gen y_recpi_pc = ((sfr * eqi) / totpop) * 1000
gen y_growth_pc = (growth / totpop) * 1000
gen y_cdfi_pc = (num_cdfi / totpop) * 1000

* Binary indicators
gen y_had_sfr = (sfr > 0)
gen y_had_growth = (growth > 0)

* Labor market outcomes
gen y_logwage = log(wage)
gen y_empop = (emp / totpop) * 100
gen y_logest = (estab / totpop) * 1000
gen y_est_pc = (estab / totpop) * 1000


/* --------------------------------------------------------------------------
   5.8 Winsorize Outcomes to Trim Outliers
-------------------------------------------------------------------------- */

hashsort id time
gstats winsor y_sfr_pc, by(year) cut(0 99) replace trim

* Drop ZIPs with any missing outcome
by id: gegen max_big = max(y_sfr_pc == .)
drop if max_big
drop max_big


/* --------------------------------------------------------------------------
   5.9 Create Time-Invariant Covariates
-------------------------------------------------------------------------- */

gen X_aland10 = log(aland10)
replace X_totpop08_12 = log(X_totpop08_12)

* State fixed effects
quietly tab cluster_state, gen(X_state)


/* --------------------------------------------------------------------------
   5.10 Create Time-Varying Covariates
-------------------------------------------------------------------------- */

gen V_totpop = log(totpop)
gen V_lenders_pc = (lenders_pc / totpop) * 1000
gen V_areadev = prop_dev_openspace + prop_dev_lowintensity + ///
                prop_dev_medintensity + prop_dev_hiintensity


/* --------------------------------------------------------------------------
   5.11 Expand Panel: Year-Specific Variables (1988-2016)
   
   Description: Creates baseline and lagged values for each year
   Purpose: Used in difference-in-differences and event study specifications
-------------------------------------------------------------------------- */

hashsort id time

* Time-varying controls by year
forvalues i = 1988/2016 {
    foreach v of varlist V_totpop V_lenders_pc V_areadev {
        gen temp = `v' if time == `i'
        by id: gegen `v'_`i' = firstnm(temp)
        drop temp
    }
}

* Entrepreneurship outcomes by year
forvalues i = 1988/2016 {
    foreach v of varlist y_sfr_pc y_eqi y_growth y_recpi_pc {
        hashsort id time
        gen temp = `v' if time == `i'
        by id: gegen Wy_`v'_`i' = firstnm(temp)
        drop temp
    }
}

* Labor market outcomes by year (available 1994-2016)
forvalues i = 1994/2016 {
    foreach v of varlist y_logwage y_empop y_logest {
        hashsort id time
        gen temp = `v' if time == `i'
        by id: gegen Wy_`v'_`i' = firstnm(temp)
        drop temp
    }
}


/* --------------------------------------------------------------------------
   5.12 Final Dataset Organization
-------------------------------------------------------------------------- */

keep id time group cluster_* i_* y_* X_* V_* Wy_*
hashsort id time
order id time group cluster_* i_* y_* X_* V_* Wy_*


********************************************************************************
***                                                                          ***
***                      END OF DATA CONSTRUCTION                           ***
***                                                                          ***
********************************************************************************

* Save final analysis dataset
save "${analysis}/final_analysis_dataset.dta", replace
export delimited "${analysis}/final_analysis_dataset.csv", replace

* Display summary
describe, short
summarize y_sfr_pc y_eqi y_growth if inrange(time, 1996, 2014)

display as text _n "=========================================="
display as text "DATA CONSTRUCTION COMPLETE!"
display as text "=========================================="
display as result _n "Final dataset saved to:"
display as result "  ${analysis}/final_analysis_dataset.dta"
display as text _n "Summary statistics:"
display as text "  Observations: " as result _N
display as text "  Time period: 1988-2016"
display as text "  Sample period for analysis: 1996-2014"
display as text _n "=========================================="
