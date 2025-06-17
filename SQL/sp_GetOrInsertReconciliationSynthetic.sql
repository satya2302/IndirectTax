-- Stored Procedure: sp_GetOrInsertReconciliationSynthetic
-- This procedure will insert one or more records using BULKINSERT and always return all records

CREATE PROCEDURE sp_GetOrInsertReconciliationSynthetic
    @Action NVARCHAR(10),
    @BulkData dbo.ReconcillationSyntheticTableType READONLY
AS
BEGIN
    SET NOCOUNT ON;

    IF @Action = 'BULKINSERT' AND EXISTS (SELECT 1 FROM @BulkData)
    BEGIN
        DELETE FROM dbo.ReconcillationSynthetic;
        INSERT INTO dbo.ReconcillationSynthetic (Region, City, County, Entity, Gross, Taxable, UnreportedTax, TaxRate, Year, Month)
        SELECT Region, City, County, Entity, Gross, Taxable, UnreportedTax, TaxRate, Year, Month FROM @BulkData;
    END
    
    -- Always return all records
    SELECT Region, City, County, Entity, Gross, Taxable, UnreportedTax, TaxRate, Year, Month
    FROM dbo.ReconcillationSynthetic;
END
