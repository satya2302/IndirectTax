import { Routes } from '@angular/router';
import { PredictComponent } from './usage/predict.component';

export const routes: Routes = [
  { path: '', redirectTo: 'usage', pathMatch: 'full' },
  { path: 'usage', loadComponent: () => import('./usage/usage.component').then(m => m.UsageComponent) },
  { path: 'predict', loadComponent: () => import('./usage/predict.component').then(m => m.PredictComponent) },
  { path: 'imports', loadComponent: () => import('./imports/imports.component').then(m => m.ImportsComponent) },
  { path: 'taxreturnsummary', loadComponent: () => import('./taxreturnsummary/taxreturnsummary.component').then(m => m.TaxReturnSummaryComponent) },
  { path: 'journals', loadComponent: () => import('./journals/journals.component').then(m => m.JournalsComponent) },
];
