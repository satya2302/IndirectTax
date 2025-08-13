using System.Collections.Generic;
using System.Data;
using Microsoft.Data.SqlClient;
using IndirectTax.Models;
using Microsoft.Extensions.Configuration;

namespace IndirectTax.Data
{
    public class ReconcillationSyntheticRepository
    {
        private readonly string _connectionString;

        public ReconcillationSyntheticRepository(IConfiguration configuration)
        {
            _connectionString = configuration.GetConnectionString("DefaultConnection")!;
        }

        public void Add(ReconcillationSynthetic item)
        {
            using (var connection = new SqlConnection(_connectionString))
            {
                connection.Open();
                using (var command = new SqlCommand(@"INSERT INTO dbo.ReconcillationSynthetic (Region, City, County, Entity, Gross, Taxable, UnreportedTax, TaxRate, Year, Month) VALUES (@Region, @City, @County, @Entity, @Gross, @Taxable, @UnreportedTax, @TaxRate, @Year, @Month)", connection))
                {
                    command.Parameters.AddWithValue("@Region", (object?)item.Region ?? DBNull.Value);
                    command.Parameters.AddWithValue("@City", (object?)item.City ?? DBNull.Value);
                    command.Parameters.AddWithValue("@County", (object?)item.County ?? DBNull.Value);
                    command.Parameters.AddWithValue("@Entity", (object?)item.Entity ?? DBNull.Value);
                    command.Parameters.AddWithValue("@Gross", item.Gross);
                    command.Parameters.AddWithValue("@Taxable", item.Taxable);
                    command.Parameters.AddWithValue("@UnreportedTax", item.UnreportedTax);
                    command.Parameters.AddWithValue("@TaxRate", item.TaxRate);
                    command.Parameters.AddWithValue("@Year", item.Year);
                    command.Parameters.AddWithValue("@Month", item.Month);
                    command.ExecuteNonQuery();
                }
            }
        }

        private const string ColRegion = "Region";
        private const string ColCity = "City";
        private const string ColCounty = "County";
        private const string ColEntity = "Entity";
        private const string ColGross = "Gross";
        private const string ColTaxable = "Taxable";
        private const string ColUnreportedTax = "UnreportedTax";
        private const string ColTaxRate = "TaxRate";
        private const string ColYear = "Year";
        private const string ColMonth = "Month";

        public IEnumerable<ReconcillationSynthetic> GetAll()
        {
            var result = new List<ReconcillationSynthetic>();
            using (var connection = new SqlConnection(_connectionString))
            {
                connection.Open();
                using (var command = new SqlCommand("sp_GetOrInsertReconciliationSynthetic", connection))
                {
                    command.CommandType = CommandType.StoredProcedure;
                    command.Parameters.AddWithValue("@Action", "GET");
                    // Pass empty DataTable for TVP
                    var table = new DataTable();
                    table.Columns.Add(ColRegion, typeof(string));
                    table.Columns.Add(ColCity, typeof(string));
                    table.Columns.Add(ColCounty, typeof(string));
                    table.Columns.Add(ColEntity, typeof(string));
                    table.Columns.Add(ColGross, typeof(decimal));
                    table.Columns.Add(ColTaxable, typeof(decimal));
                    table.Columns.Add(ColUnreportedTax, typeof(decimal));
                    table.Columns.Add(ColTaxRate, typeof(decimal));
                    table.Columns.Add(ColYear, typeof(int));
                    table.Columns.Add(ColMonth, typeof(int));
                    var tvpParam = command.Parameters.AddWithValue("@BulkData", table);
                    tvpParam.SqlDbType = SqlDbType.Structured;
                    tvpParam.TypeName = "dbo.ReconcillationSyntheticTableType";
                    using (var reader = command.ExecuteReader())
                    {
                        while (reader.Read())
                        {
                            result.Add(new ReconcillationSynthetic
                            {
                                Region = reader[ColRegion] == DBNull.Value ? string.Empty : reader.GetString(reader.GetOrdinal(ColRegion)),
                                City = reader[ColCity] == DBNull.Value ? string.Empty : reader.GetString(reader.GetOrdinal(ColCity)),
                                County = reader[ColCounty] == DBNull.Value ? string.Empty : reader.GetString(reader.GetOrdinal(ColCounty)),
                                Entity = reader[ColEntity] == DBNull.Value ? string.Empty : reader.GetString(reader.GetOrdinal(ColEntity)),
                                Gross = reader.GetDecimal(reader.GetOrdinal(ColGross)),
                                Taxable = reader.GetDecimal(reader.GetOrdinal(ColTaxable)),
                                UnreportedTax = reader.GetDecimal(reader.GetOrdinal(ColUnreportedTax)),
                                TaxRate = reader.GetDecimal(reader.GetOrdinal(ColTaxRate)),
                                Year = reader.GetInt32(reader.GetOrdinal(ColYear)),
                                Month = reader.GetInt32(reader.GetOrdinal(ColMonth))
                            });
                        }
                    }
                }
            }
            return result;
        }
        public void Update(ReconcillationUpdate items)
        {
              using (var connection = new SqlConnection(_connectionString))
            {
                connection.Open();
                using (var command = new SqlCommand("sp_updateReconcillation", connection))
                {
                    command.CommandType = CommandType.StoredProcedure;
                    command.Parameters.AddWithValue("@Action", "BULKINSERT");
                    // Always use TVP, even for single item
                    
                }
                
            }
        }
        public IEnumerable<ReconcillationSynthetic> BulkAddAndGetAll(IEnumerable<ReconcillationSynthetic> items)
        {
            var result = new List<ReconcillationSynthetic>();
            using (var connection = new SqlConnection(_connectionString))
            {
                connection.Open();
                using (var command = new SqlCommand("sp_GetOrInsertReconciliationSynthetic", connection))
                {
                    command.CommandType = CommandType.StoredProcedure;
                    command.Parameters.AddWithValue("@Action", "BULKINSERT");
                    // Always use TVP, even for single item
                    var table = new DataTable();
                    table.Columns.Add(ColRegion, typeof(string));
                    table.Columns.Add(ColCity, typeof(string));
                    table.Columns.Add(ColCounty, typeof(string));
                    table.Columns.Add(ColEntity, typeof(string));
                    table.Columns.Add(ColGross, typeof(decimal));
                    table.Columns.Add(ColTaxable, typeof(decimal));
                    table.Columns.Add(ColUnreportedTax, typeof(decimal));
                    table.Columns.Add(ColTaxRate, typeof(decimal));
                    table.Columns.Add(ColYear, typeof(int));
                    table.Columns.Add(ColMonth, typeof(int));
                    foreach (var item in items)
                    {
                        table.Rows.Add(
                            (object?)item.Region ?? DBNull.Value,
                            (object?)item.City ?? DBNull.Value,
                            (object?)item.County ?? DBNull.Value,
                            (object?)item.Entity ?? DBNull.Value,
                            item.Gross,
                            item.Taxable,
                            item.UnreportedTax,
                            item.TaxRate,
                            item.Year,
                            item.Month
                        );
                    }
                    var tvpParam = command.Parameters.AddWithValue("@BulkData", table);
                    tvpParam.SqlDbType = SqlDbType.Structured;
                    tvpParam.TypeName = "dbo.ReconcillationSyntheticTableType";
                    using (var reader = command.ExecuteReader())
                    {
                        while (reader.Read())
                        {
                            result.Add(new ReconcillationSynthetic
                            {
                                Region = reader[ColRegion] == DBNull.Value ? string.Empty : reader.GetString(reader.GetOrdinal(ColRegion)),
                                City = reader[ColCity] == DBNull.Value ? string.Empty : reader.GetString(reader.GetOrdinal(ColCity)),
                                County = reader[ColCounty] == DBNull.Value ? string.Empty : reader.GetString(reader.GetOrdinal(ColCounty)),
                                Entity = reader[ColEntity] == DBNull.Value ? string.Empty : reader.GetString(reader.GetOrdinal(ColEntity)),
                                Gross = reader.GetDecimal(reader.GetOrdinal(ColGross)),
                                Taxable = reader.GetDecimal(reader.GetOrdinal(ColTaxable)),
                                UnreportedTax = reader.GetDecimal(reader.GetOrdinal(ColUnreportedTax)),
                                TaxRate = reader.GetDecimal(reader.GetOrdinal(ColTaxRate)),
                                Year = reader.GetInt32(reader.GetOrdinal(ColYear)),
                                Month = reader.GetInt32(reader.GetOrdinal(ColMonth))
                            });
                        }
                    }
                }
            }
            return result;
        }
    }
}
