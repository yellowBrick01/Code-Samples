--Base Tables: Stock, Client, Fund_Family

--CREATE object for clients
CREATE OR REPLACE TYPE address_ty AS OBJECT(
 streetName VARCHAR2(50),
 cityName VARCHAR2(20),
 zipCode VARCHAR2(5)
);

--Create Table Client
CREATE TABLE Client
(TaxID VARCHAR2(10) NOT NULL, 
cAddress address_ty NOT NULL, 
cType VARCHAR2(1) CHECK (cType IN ('I', 'B')),
CONSTRAINT client_taxID_pk PRIMARY KEY (taxID));

--Insert Values into Client table(I) 
INSERT INTO Client values
('17-00-9947',address_ty('63 International St','New York','10018'), 'I');
INSERT INTO Client values
('30-0058-21',address_ty('8 Welch Court','Los Angeles','90018'), 'I');
INSERT INTO Client values
('67-2764-70',address_ty('923 Summit Plaza','Chicago','60654'), 'I');
INSERT INTO Client values
('25-4236-78',address_ty('7987 W 5th St','San Francisco','94105'), 'I');
INSERT INTO Client values
('45-8567-16',address_ty('0479 Kinsman Way','Seattle','98119'), 'I');

--insert values into Client(B)
INSERT INTO Client values
('00-00-0000', address_ty('00', '00', '00'),'B');
INSERT INTO Client values
('24-48-3691', address_ty('85 Willow RD', 'Menlo Park', '94025'), 'B');
INSERT INTO Client values
('51-68-7891', address_ty('44 Wall St Suite 501', 'New York', '10005'), 'B');
INSERT INTO Client values
('11-23-5813', address_ty('2 N Lake Ave #100', 'Pasadena', '91101'), 'B');
INSERT INTO Client values
('36-24-0918', address_ty('11713 Gorham Ave', 'Los Angeles', '90049'), 'B');
INSERT INTO Client values
('25-10-1020', address_ty('123 S Lake Ave', 'Pasadena', '91101'), 'B');

--CREATE object for individual clients
CREATE OR REPLACE TYPE name_ty AS OBJECT(
 indvFirstName VARCHAR2(20),
 indvLastName VARCHAR2(20)
);
CREATE OR REPLACE TYPE personal_info_ty as OBJECT(
 indvDOB DATE,
 indvGender VARCHAR2(1)
);

--CREATE TABLE for individual clients
CREATE TABLE Individual
(taxID VARCHAR2 (10) NOT NULL,
indvName name_ty,--we can add NOT NULL if we want
indvInformation personal_info_ty CHECK (indvInformation.indvGender IN ('M', 'F', 'O')),
CONSTRAINT individual_taxID_pk PRIMARY KEY (taxID));

--insert value into individual table
INSERT INTO Individual values
('17-00-9947', name_ty('Tom','Brumble'), personal_info_ty('13-APR-1969', 'M'));
INSERT INTO Individual values
('30-0058-21',name_ty('Julieta' ,'Jane'),personal_info_ty('04-DEC-1988','O'));
INSERT INTO Individual values
('67-2764-70',name_ty('Adam', 'Rosindill'), personal_info_ty('29-MAY-1997','M'));
INSERT INTO Individual values
('25-4236-78',name_ty('Sam', 'Taylor'), personal_info_ty('11-NOV-2000','M'));
INSERT INTO Individual values
('45-8567-16',name_ty('Andrea', 'Levin'), personal_info_ty('01-FEB-2003', 'F'));

--Add Foreign Keys to Individual
ALTER TABLE Individual
ADD FOREIGN KEY (taxID) REFERENCES Client(taxID);

--Create Table Business
CREATE TABLE Business
(taxID VARCHAR2 (10) NOT NULL,
 bizCompanyName VARCHAR2(40) NOT NULL,
bizEstablishedDate Date,
CONSTRAINT business_taxID_pk PRIMARY KEY (taxID));

--insert values into business table
INSERT INTO Business values
('00-00-0000', 'Fake Company',NULL);
INSERT INTO Business values
('24-48-3691', 'Robinhood', '18-APR-2013');
INSERT INTO Business values
('51-68-7891', 'Webull', '24-MAY-2017');
INSERT INTO Business values
('11-23-5813', 'TD Ameritrade', '16-MAR-1975');
INSERT INTO Business values
('36-24-0918', 'E-Treade', '18-JUL-1982');
INSERT INTO Business values
('25-10-1020', 'Fidelity', '05-MAY-1946');

--Add Foreign Keys to Business 
ALTER TABLE Business
ADD FOREIGN KEY (taxID) REFERENCES Client(taxID);

--Create Table Stock 
CREATE TABLE Stock
(stockTicker VARCHAR2(4) NOT NULL,
sName VARCHAR2(40) NOT NULL UNIQUE, 
ratings VARCHAR2(1) CHECK (ratings IN ('A', 'B', 'C', 'D', 'F')), 
prinBusiness VARCHAR2(22), 
sCurrentPrice NUMBER(10,2),
sAnnualHigh NUMBER(10,2),
sAnnualLow NUMBER(10,2),
oneYearReturn NUMBER, 
fiveYearReturn NUMBER,
CONSTRAINT stock_stockTicker_pk PRIMARY KEY (stockTicker));

--Insert into Table Stock 

insert into Stock values
('TSLA', 'Tesla, Inc.', 'A', 'Auto Manufacturers', 683.86, 880.02, 114.60, 522.69, 1266.46);

insert into Stock values
('GME', 'GameStop Corp.', 'B', 'Technology', 212.54, 212.85, 139.36, 55.48, 104.16);

insert into Stock values
('AAPL', 'Apple Inc.', 'B', 'Technology', 130.36, 139.07, 67, 94.57, 374.73);

insert into Stock values
('AMZN', 'Amazon.com, Inc.', 'B', 'Consumer Cyclical', 3299.30, 3401.80, 2042.76, 61.51, 427.14);

insert into Stock values
('FB', 'Facebook, Inc.', 'B', 'Communication Services', 313.02, 313.02, 175.19, 16.23, 189.02);

insert into Stock values
('BABA', 'Alibaba Group Holding Limited', 'B', 'Consumer Cyclical', 228.24, 309.92, 194.48, 522.69, 1266.46);

--Create Table Fund_Family 
CREATE TABLE Fund_Family
(fFamilyID VARCHAR2(10) NOT NULL, 
 mFundName VARCHAR2(40),
familyAddress address_ty);

--Add PK to Fund Family TABLE
ALTER TABLE Fund_Family
ADD CONSTRAINT fund_family_pk PRIMARY KEY(fFamilyID);

--Insert into Table Fund_Family 
--insert into Fund_Family values('JMVYX','JPMorgan', '1111 Polaris Parkway', 'Columbus', 'OH 43240')  --- > Should we add a column for state? or its fine.
insert into Fund_Family values
('HLIEX25','JPMorgan', address_ty('245 Park Avenue', 'New York', '10167'));

insert into Fund_Family values
('RPMGX44','T. Rowe Price', address_ty('100 East Pratt Street', 'Baltimore', '21202'));

insert into Fund_Family values
('FXAIX01','Fidelity Investments', address_ty('82 Devonshire Street', 'Boston', '02109'));

insert into Fund_Family values
('JMVYX95','JPMorgan', address_ty('1111 Polaris Parkway', 'Columbus', '43240'));

insert into Fund_Family values
('VFIAX62','Vanguard', address_ty('PO Box 2600', 'Valley Forge', '19482'));

--Create Table Mutual_Fund
CREATE TABLE Mutual_Fund
(fundTicker VARCHAR2(10) NOT NULL, 
 mFundName VARCHAR2(40),
 ratings VARCHAR2(1) CHECK (ratings IN ('A', 'B', 'C', 'D', 'F')), 
prinObjective VARCHAR2(20), 
mFundCurrentPrice NUMBER(10,2), 
mFundAnnualHigh NUMBER(10,2), 
mFundAnnualLow NUMBER(10,2),
percentYield NUMBER(10,2),
mFundFamilyID VARCHAR2(10),
CONSTRAINT mutual_fund_fundTicker_pk PRIMARY KEY (fundTicker));

--Insert into Table Mutual_Fund 
Insert Into Mutual_Fund values
('FXAIX','Fidelity 500 Index Fund', 'A','Large Blend', 142.59, 141.96, 96.74, 1.57, 'FXAIX01');

Insert Into Mutual_Fund values
('HLIEX','JPMorgan Equity Income Fund Class I', 'B','Large Value', 21.83, 21.83, 15.95, 1.63, 'HLIEX25');

Insert Into Mutual_Fund values
('VFIAX','Vanguard 500 Index Fund Admiral Shares', 'C','Large Blend', 378.32, 376.64, 257.68, 1.47, 'VFIAX62');

Insert Into Mutual_Fund values
('RPMGX','T. Rowe Price Mid-Cap Growth Fund','B','Mid-Cap Growth', 119.71, 118.68, 79.73, 0.00, 'RPMGX44');

Insert Into Mutual_Fund values
('JMVYX','JPMorgan Mid Cap Value Fund Class R6','D','Mid-Cap Value', 43.39, 43.45, 28.52, 1.02, 'JMVYX95');

--Create Foreign Key in Mutual_Fund 
ALTER TABLE Mutual_Fund
ADD FOREIGN KEY (mFundFamilyID) REFERENCES Fund_Family(fFamilyID);

--Create Table Stock_Portfolio 
CREATE TABLE Stock_Portfolio
(taxID VARCHAR2 (10) NOT NULL, 
 stockTicker VARCHAR2(5) NOT NULL,
 sNumberOfShares NUMBER,
CONSTRAINT stock_portfolio_ck_pk PRIMARY KEY (taxID, stockTicker));

--Insert into Table Stock_Portfolio 

Insert Into Stock_Portfolio values
('45-8567-16','GME','600');

Insert Into Stock_Portfolio values
('25-4236-78','GME','800');


Insert Into Stock_Portfolio values
('67-2764-70','BABA','85');

Insert Into Stock_Portfolio values
('45-8567-16','AAPL','200');

Insert Into Stock_Portfolio values
('25-4236-78','AMZN','100');

Insert Into Stock_Portfolio values
('30-0058-21','BABA','300');

Insert Into Stock_Portfolio values
('30-0058-21','FB','700');

Insert Into Stock_Portfolio values
('17-00-9947','TSLA','1200');

--Add Foreign Keys to Stock_Portfolio
ALTER TABLE Stock_Portfolio
ADD FOREIGN KEY (taxID) REFERENCES Client(taxID); 
ALTER TABLE Stock_Portfolio
ADD FOREIGN KEY (stockTicker) REFERENCES Stock(stockTicker);

--Create Table Mutual_Fund_Portfolio
CREATE TABLE Mutual_Fund_Portfolio
(taxID VARCHAR2 (10) NOT NULL,
 fundTicker VARCHAR2(5) NOT NULL,
 mFundNumberOfShares NUMBER,
CONSTRAINT mutual_fund_portfolio_ck_pk PRIMARY KEY (taxID, fundTicker));

--Insert into Table Mutual_Fund_Portfolio 

Insert Into Mutual_Fund_Portfolio values
('25-4236-78','VFIAX','700');


Insert Into Mutual_Fund_Portfolio values
('17-00-9947','JMVYX','100');

Insert Into Mutual_Fund_Portfolio values
('25-4236-78','RPMGX','100');

Insert Into Mutual_Fund_Portfolio values
('17-00-9947','HLIEX','600');

Insert Into Mutual_Fund_Portfolio values
('67-2764-70','RPMGX','200');

Insert Into Mutual_Fund_Portfolio values
('30-0058-21','VFIAX','1300');

Insert Into Mutual_Fund_Portfolio values
('67-2764-70','FXAIX','500');

Insert Into Mutual_Fund_Portfolio values
('30-0058-21','HLIEX','400');

Insert Into Mutual_Fund_Portfolio values
('45-8567-16','FXAIX','800');

--Add Foreign Keys to Mutual_Fund_Portfolio
ALTER TABLE Mutual_Fund_Portfolio
ADD FOREIGN KEY (taxID) REFERENCES Client(taxID); 
ALTER TABLE Mutual_Fund_Portfolio
ADD FOREIGN KEY (fundTicker) REFERENCES Mutual_Fund(fundTicker);