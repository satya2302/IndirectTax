CREATE TYPE dbo.JournalSyntheticTableType AS TABLE
(
    Region NVARCHAR(100) NULL,
    City NVARCHAR(100) NULL,
    County NVARCHAR(100) NULL,
    Entity NVARCHAR(100) NULL,
    District NVARCHAR(100) NULL,
    Gross DECIMAL(18,2) NULL,
    Taxable DECIMAL(18,2) NULL,
    InputSource NVARCHAR(100) NULL,
    PlaceDetermination NVARCHAR(100) NULL,
    EntryId INT NULL,
    Id NVARCHAR(100) NULL,
    LogId NVARCHAR(100) NULL,
    TaxRate DECIMAL(5,4) NULL,
    Year INT NULL,
    Month INT NULL
)
