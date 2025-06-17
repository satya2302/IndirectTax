using System.Collections.Generic;
using System.Data;
using Microsoft.Data.SqlClient;
using IndirectTax.Models;
using Microsoft.Extensions.Configuration;

namespace IndirectTax.Data
{
    public class JournalSyntheticRepository
    {
        private readonly string _connectionString;

        public JournalSyntheticRepository(IConfiguration configuration)
        {
            _connectionString = configuration.GetConnectionString("DefaultConnection")!;
        }

        public IEnumerable<JournalSynthetic> BulkAddAndGetAll(IEnumerable<JournalSynthetic> items)
        {
            using (var connection = new SqlConnection(_connectionString))
            using (var command = new SqlCommand("sp_GetOrInsertJournalSynthetic", connection))
            {
                command.CommandType = CommandType.StoredProcedure;
                command.Parameters.AddWithValue("@Action", "BULKINSERT");
                command.Parameters.AddWithValue("@BulkData", ToDataTable(items));
                connection.Open();
                using (var reader = command.ExecuteReader())
                {
                    var result = new List<JournalSynthetic>();
                    while (reader.Read())
                    {
                        result.Add(new JournalSynthetic
                        {
                            Region = reader["Region"] as string,
                            City = reader["City"] as string,
                            County = reader["County"] as string,
                            Entity = reader["Entity"] as string,
                            District = reader["District"] as string,
                            Gross = reader.GetDecimal(reader.GetOrdinal("Gross")),
                            Taxable = reader.GetDecimal(reader.GetOrdinal("Taxable")),
                            InputSource = reader["InputSource"] as string,
                            PlaceDetermination = reader["PlaceDetermination"] as string,
                            EntryId = reader["EntryId"] == DBNull.Value ? 0 : Convert.ToInt32(reader["EntryId"]),
                            Id = reader["Id"] as string,
                            LogId = reader["LogId"] as string,
                            TaxRate = reader.GetDecimal(reader.GetOrdinal("TaxRate")),
                            Year = reader.GetInt32(reader.GetOrdinal("Year")),
                            Month = reader.GetInt32(reader.GetOrdinal("Month")),
                        });
                    }
                    return result;
                }
            }
        }

        public IEnumerable<JournalSynthetic> GetAll()
        {
            using (var connection = new SqlConnection(_connectionString))
            using (var command = new SqlCommand("sp_GetOrInsertJournalSynthetic", connection))
            {
                command.CommandType = CommandType.StoredProcedure;
                command.Parameters.AddWithValue("@Action", "GET");
                command.Parameters.AddWithValue("@BulkData", ToDataTable(new List<JournalSynthetic>()));
                connection.Open();
                using (var reader = command.ExecuteReader())
                {
                    var result = new List<JournalSynthetic>();
                    while (reader.Read())
                    {
                        result.Add(new JournalSynthetic
                        {
                            Region = reader["Region"] as string,
                            City = reader["City"] as string,
                            County = reader["County"] as string,
                            Entity = reader["Entity"] as string,
                            District = reader["District"] as string,
                            Gross = reader.GetDecimal(reader.GetOrdinal("Gross")),
                            Taxable = reader.GetDecimal(reader.GetOrdinal("Taxable")),
                            InputSource = reader["InputSource"] as string,
                            PlaceDetermination = reader["PlaceDetermination"] as string,
                            EntryId = reader["EntryId"] == DBNull.Value ? 0 : Convert.ToInt32(reader["EntryId"]),
                            Id = reader["Id"] as string,
                            LogId = reader["LogId"] as string,
                            TaxRate = reader.GetDecimal(reader.GetOrdinal("TaxRate")),
                            Year = reader.GetInt32(reader.GetOrdinal("Year")),
                            Month = reader.GetInt32(reader.GetOrdinal("Month")),
                        });
                    }
                    return result;
                }
            }
        }

        private DataTable ToDataTable(IEnumerable<JournalSynthetic> items)
        {
            var table = new DataTable();
            table.Columns.Add("Region", typeof(string));
            table.Columns.Add("City", typeof(string));
            table.Columns.Add("County", typeof(string));
            table.Columns.Add("Entity", typeof(string));
            table.Columns.Add("District", typeof(string));
            table.Columns.Add("Gross", typeof(decimal));
            table.Columns.Add("Taxable", typeof(decimal));
            table.Columns.Add("InputSource", typeof(string));
            table.Columns.Add("PlaceDetermination", typeof(string));
            table.Columns.Add("EntryId", typeof(int));
            table.Columns.Add("Id", typeof(string));
            table.Columns.Add("LogId", typeof(string));
            table.Columns.Add("TaxRate", typeof(decimal));
            table.Columns.Add("Year", typeof(int));
            table.Columns.Add("Month", typeof(int));
            foreach (var item in items)
            {
                table.Rows.Add(
                    item.Region ?? (object)DBNull.Value,
                    item.City ?? (object)DBNull.Value,
                    item.County ?? (object)DBNull.Value,
                    item.Entity ?? (object)DBNull.Value,
                    item.District ?? (object)DBNull.Value,
                    item.Gross,
                    item.Taxable,
                    item.InputSource ?? (object)DBNull.Value,
                    item.PlaceDetermination ?? (object)DBNull.Value,
                    item.EntryId,
                    item.Id ?? (object)DBNull.Value,
                    item.LogId ?? (object)DBNull.Value,
                    item.TaxRate,
                    item.Year,
                    item.Month
                );
            }
            return table;
        }
    }
}
