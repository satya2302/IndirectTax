# BackEnd Folder Technical Specification

## Overview
The `BackEnd` folder contains the ASP.NET Core backend for the IndirectTax project. It provides RESTful APIs for data access, business logic, and integration with the frontend and machine learning components.

## Directory Structure
```
BackEnd/
  appsettings.Development.json
  appsettings.json
  BackEnd.csproj
  BackEnd.http
  BackEnd.sln
  Program.cs
  README.MD
  Controllers/
    UsageController.cs
  Data/
    DataUsageRepository.cs
  Models/
    DataUsage.cs
  Properties/
    launchSettings.json
```

## Key Contents
- **Configuration Files**: `appsettings.json`, `appsettings.Development.json` for environment and application settings.
- **Project Files**: `BackEnd.csproj`, `BackEnd.sln` for .NET project and solution management.
- **Entry Point**: `Program.cs` initializes and runs the web application.
- **Controllers**: API endpoints (e.g., `UsageController.cs`) for handling HTTP requests.
- **Data Layer**: Repository classes (e.g., `DataUsageRepository.cs`) for data access and persistence.
- **Models**: Data models (e.g., `DataUsage.cs`) representing business entities.
- **Properties**: Launch settings for development and debugging.

## Features
- RESTful API endpoints for data usage and other business logic.
- Repository pattern for data access abstraction.
- Model binding and validation.
- Environment-based configuration.
- Integration-ready for frontend and ML components.

## Dependencies
- .NET 6.0 or later (see `BackEnd.csproj` for details)
- ASP.NET Core
- Entity Framework Core (if used)
- Other NuGet packages as specified in the project file

Install dependencies with:
```
dotnet restore
```

## Usage
- Run the backend server:
  ```
  dotnet run
  ```
- The API will be available at the URL specified in `launchSettings.json` (typically `https://localhost:5001` or similar).
- Test endpoints using tools like Postman or the provided `BackEnd.http` file.

## Notes
- Update configuration in `appsettings.json` as needed for database connections or environment variables.
- Add new controllers for additional API endpoints.
- Use the repository and model structure for maintainable, testable code.
