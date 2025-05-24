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
	[EFilings] [numeric](18, 0) NULL,
	[Users_Transactions] [numeric](18, 0) NULL,
	[Users_Returns] [numeric](18, 0) NULL,
	[Users_efile] [numeric](18, 0) NULL
) ON [PRIMARY]
GO


