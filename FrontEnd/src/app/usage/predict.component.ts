import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-predict',
  standalone: true,
  imports: [CommonModule, HttpClientModule],
  templateUrl: './predict.component.html',
  styleUrl: './predict.component.css'
})
export class PredictComponent implements OnInit {
  predictions: any[] = [];
  loading = false;
  error = '';

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.fetchPredictions();
  }

  fetchPredictions() {
    this.loading = true;
    this.http.get<any[]>('http://127.0.0.1:5000/predict').subscribe({
      next: (data) => {
        this.predictions = data;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to load predictions.';
        this.loading = false;
      }
    });
  }
}
