import { Component } from '@angular/core';

@Component({
  selector: 'app-usage-dashboard',
  standalone: true,
  template: `
    <div class="dashboard-container">
      <h2>Usage Dashboard</h2>
      <div class="dashboard-cards">
        <div class="dashboard-card">
          <div class="dashboard-title">Total Transactions</div>
          <div class="dashboard-value">12,345</div>
        </div>
        <div class="dashboard-card">
          <div class="dashboard-title">Total Tax Returns</div>
          <div class="dashboard-value">2,345</div>
        </div>
        <div class="dashboard-card">
          <div class="dashboard-title">Total EFilings</div>
          <div class="dashboard-value">1,234</div>
        </div>
        <div class="dashboard-card">
          <div class="dashboard-title">Active Users</div>
          <div class="dashboard-value">56</div>
        </div>
      </div>
      <div class="dashboard-summary">
        <h3>Summary</h3>
        <ul>
          <li>Usage is up 15% compared to last month.</li>
          <li>Tax return submissions are steady.</li>
          <li>EFilings have increased by 8%.</li>
          <li>Active users remain consistent.</li>
        </ul>
      </div>
    </div>
  `,
  styleUrl: './usage-dashboard.component.css'
})
export class UsageDashboardComponent {}
