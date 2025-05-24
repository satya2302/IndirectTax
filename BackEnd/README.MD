# IndirectTax Backend API

This is a .NET 8 Web API project for the Indirect Tax system. It includes a UsageController that connects to a local SQL Server 'IndirectTax' database and reads data from the 'Datausage' table using ADO.NET.

## Getting Started

1. Ensure you have .NET 8 SDK installed.
2. Update your connection string in `appsettings.json` to point to your local SQL Server and the 'IndirectTax' database.
3. Build and run the project:
   ```powershell
   dotnet build
   dotnet run
   ```

## Project Structure
- `Controllers/UsageController.cs`: API endpoint for usage data.
- `Models/DataUsage.cs`: Model for usage data.
- `Data/DataUsageRepository.cs`: Data access using ADO.NET.

## Notes
- Make sure your SQL Server is running and accessible.
- The default connection string assumes SQL Server is on localhost.
