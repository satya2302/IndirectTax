# FrontEnd Folder Technical Specification

## Overview
The `FrontEnd` folder contains the Angular-based user interface for the IndirectTax project. It provides interactive components for data visualization, user input, and communication with the backend services.

## Directory Structure
```
FrontEnd/
  angular.json
  package.json
  README.md
  tsconfig.app.json
  tsconfig.json
  tsconfig.spec.json
  public/
    favicon.ico
  src/
    index.html
    main.ts
    styles.css
    app/
      app.component.css
      app.component.html
      app.component.spec.ts
      app.component.ts
      app.config.ts
      app.module.ts
      app.routes.ts
      star.png
      chatbot/
        chatbot.component.css
        chatbot.component.html
        chatbot.component.ts
        index.ts
      imports/
        imports.component.css
        imports.component.html
        imports.component.ts
      journals/
        journals.component.css
        journals.component.html
        journals.component.ts
      menu/
        menu.component.css
        menu.component.html
        menu.component.spec.ts
        menu.component.ts
      taxreturnsummary/
        taxreturnsummary.component.css
        taxreturnsummary.component.html
        taxreturnsummary.component.ts
      usage/
        data-usage.model.ts
        predict.component.css
        predict.component.html
        predict.component.ts
        usage.component.css
        usage.component.html
        usage.component.spec.ts
        usage.component.ts
        usage.service.ts
```

## Key Contents
- **Configuration Files**: `angular.json`, `tsconfig*.json`, `package.json` for project, build, and dependency management.
- **Public Assets**: Favicon and other static files in `public/`.
- **Source Code**: All application logic and UI in `src/`.
  - **app/**: Main application module and root components.
    - **chatbot/**: Chatbot UI and logic.
    - **imports/**: Handles import-related UI.
    - **journals/**: Journal data display and interaction.
    - **menu/**: Navigation menu components.
    - **taxreturnsummary/**: Tax return summary display.
    - **usage/**: Data usage, prediction, and related services.

## Features
- Modern Angular SPA architecture.
- Modular component structure for maintainability.
- Data visualization and user interaction for tax, journal, and usage data.
- Chatbot interface for user assistance.
- Service integration for backend API communication (e.g., `usage.service.ts`).
- Responsive design with CSS modules.
- Routing and navigation between app sections.

## Dependencies
- Angular (see `package.json` for version)
- RxJS
- TypeScript
- Other dependencies as listed in `package.json`

Install dependencies with:
```
npm install
```

## Usage
- Start the development server:
  ```
  npm start
  ```
- Build for production:
  ```
  npm run build
  ```
- Run tests:
  ```
  npm test
  ```

## Notes
- All source code is under `src/app/` and organized by feature.
- Update dependencies via `npm install <package>` as needed.
- For backend API endpoints, configure URLs in the relevant service files.
