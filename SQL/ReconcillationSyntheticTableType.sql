CREATE TYPE dbo.ReconcillationSyntheticTableType AS TABLE
(
    Region NVARCHAR(100),
    City NVARCHAR(100),
    County NVARCHAR(100),
    Entity NVARCHAR(100),
    Gross DECIMAL(18,2),
    Taxable DECIMAL(18,2),
    UnreportedTax DECIMAL(18,2),
    TaxRate DECIMAL(18,2),
    Year INT,
    Month INT
);
