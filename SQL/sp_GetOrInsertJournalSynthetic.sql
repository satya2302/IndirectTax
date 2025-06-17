-- Stored Procedure: sp_GetOrInsertJournalSynthetic
-- This procedure will insert one or more records using BULKINSERT and always return all records

CREATE PROCEDURE sp_GetOrInsertJournalSynthetic
    @Action NVARCHAR(10),
    @BulkData dbo.JournalSyntheticTableType READONLY
AS
BEGIN
    SET NOCOUNT ON;

    IF @Action = 'BULKINSERT' AND EXISTS (SELECT 1 FROM @BulkData)
    BEGIN
        DELETE FROM dbo.JournalSynthetic;
        INSERT INTO dbo.JournalSynthetic (Region, City, County, Entity, District, Gross, Taxable, InputSource, PlaceDetermination, EntryId, Id, LogId, TaxRate, Year, Month)
        SELECT Region, City, County, Entity, District, Gross, Taxable, InputSource, PlaceDetermination, EntryId, Id, LogId, TaxRate, Year, Month FROM @BulkData;
    END
    
    -- Always return all records
    SELECT Region, City, County, Entity, District, Gross, Taxable, InputSource, PlaceDetermination, EntryId, Id, LogId, TaxRate, Year, Month
    FROM dbo.JournalSynthetic;
END
