import { Component, OnInit } from '@angular/core';
import { UsageService } from './usage.service';
import { DataUsage } from './data-usage.model';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { Router } from '@angular/router';
import { UsageChartComponent } from './usage-chart.component';

@Component({
  selector: 'app-usage',
  standalone: true,
  imports: [CommonModule, HttpClientModule, UsageChartComponent],
  templateUrl: './usage.component.html',
  styleUrl: './usage.component.css',
  providers: [] // <-- Add this line if not present
})
export class UsageComponent implements OnInit {
  usageList: DataUsage[] = [];
  loading = false;
  error = '';
  activeTab: 'data' | 'graph' = 'data';

  constructor(private usageService: UsageService, private router: Router) {}

  ngOnInit() {
    this.loadChartJs();
    this.fetchUsage();
  }

  loadChartJs() {
    if (!(window as any).Chart) {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
      script.onload = () => {};
      document.body.appendChild(script);
    }
  }

  fetchUsage() {
    this.loading = true;
    this.usageService.getUsage().subscribe({
      next: (data) => {
        this.usageList = data;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to load usage data.';
        this.loading = false;
      }
    });
  }

  onPredict() {
    this.router.navigate(['/predict']);
  }

  switchTab(tab: 'data' | 'graph') {
    this.activeTab = tab;
  }
}
