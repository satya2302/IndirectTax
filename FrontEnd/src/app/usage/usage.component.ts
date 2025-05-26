import { Component, OnInit } from '@angular/core';
import { UsageService } from './usage.service';
import { DataUsage } from './data-usage.model';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { Router } from '@angular/router';

@Component({
  selector: 'app-usage',
  standalone: true,
  imports: [CommonModule, HttpClientModule],
  templateUrl: './usage.component.html',
  styleUrl: './usage.component.css',
  providers: [] // <-- Add this line if not present
})
export class UsageComponent implements OnInit {
  usageList: DataUsage[] = [];
  loading = false;
  error = '';

  constructor(private usageService: UsageService, private router: Router) {}

  ngOnInit() {
    this.fetchUsage();
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
}
