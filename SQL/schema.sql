USE [IndirectTax]
GO

/****** Object:  Table [dbo].[DataUsage]    Script Date: 5/24/2025 2:41:48 PM ******/
IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[DataUsage]') AND type in (N'U'))
DROP TABLE [dbo].[DataUsage]
GO

/****** Object:  Table [dbo].[DataUsage]    Script Date: 5/24/2025 2:41:48 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[DataUsage](
	[Year] [numeric](18, 0) NULL,
	[Month] [nchar](10) NULL,
	[Transactions] [numeric](18, 0) NULL,
	[TaxReturns] [numeric](18, 0) NULL,
	[EFilings] [numeric](18, 0) NULL
	
) ON [PRIMARY]
GO

IF EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[ReconcillationSynthetic]') AND type in (N'U'))
DROP TABLE [dbo].[ReconcillationSynthetic]
GO

SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[ReconcillationSynthetic](
    [Region] NVARCHAR(20) NULL,
    [City] NVARCHAR(50) NULL,
    [County] NVARCHAR(50) NULL,
    [Entity] NVARCHAR(50) NULL,
    [Gross] DECIMAL(18,2) NULL,
    [Taxable] DECIMAL(18,2) NULL,
    [UnreportedTax] DECIMAL(18,2) NULL,
    [TaxRate] DECIMAL(10,4) NULL,
    [Year] INT NULL,
    [Month] INT NULL
) ON [PRIMARY]
GO


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
GO

-- Create JournalSynthetic table if it does not exist
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'JournalSynthetic')
BEGIN
    CREATE TABLE [dbo].[JournalSynthetic] (
        [Region] NVARCHAR(100) NULL,
        [City] NVARCHAR(100) NULL,
        [County] NVARCHAR(100) NULL,
        [Entity] NVARCHAR(100) NULL,
        [District] NVARCHAR(100) NULL,
        [Gross] DECIMAL(18,2) NULL,
        [Taxable] DECIMAL(18,2) NULL,
        [InputSource] NVARCHAR(100) NULL,
        [PlaceDetermination] NVARCHAR(100) NULL,
        [EntryId] INT NULL,
        [Id] NVARCHAR(100) NULL,
        [LogId] NVARCHAR(100) NULL,
        [TaxRate] DECIMAL(5,4) NULL,
        [Year] INT NULL,
        [Month] INT NULL
    );
END
GO