using System.Collections.Generic;
using System.Data;
using Microsoft.Data.SqlClient;
using IndirectTax.Models;
using Microsoft.Extensions.Configuration;

namespace IndirectTax.Data
{
    public class DataUsageRepository
    {
        private readonly string _connectionString;

        public DataUsageRepository(IConfiguration configuration)
        {
            _connectionString = configuration.GetConnectionString("DefaultConnection");
        }

        public IEnumerable<DataUsage> GetAll()
        {
            var result = new List<DataUsage>();
            using (var connection = new SqlConnection(_connectionString))
            {
                connection.Open();
                using (var command = new SqlCommand("SELECT * FROM Datausage", connection))
                using (var adapter = new SqlDataAdapter(command))
                {
                    var dataTable = new DataTable();
                    adapter.Fill(dataTable);

                    foreach (DataRow row in dataTable.Rows)
                    {
                        result.Add(new DataUsage
                        {
                            year = row.Field<decimal>(0),
                            Month = row.IsNull(1) ? null : row.Field<string>(1),
                            Transactions = row.Field<decimal>(2),
                            TaxReturns = row.Field<decimal>(3),
                            EFilings = row.Field<decimal>(4)
                        });
                    }
                }
            }
            return result;
        }
    }
}
