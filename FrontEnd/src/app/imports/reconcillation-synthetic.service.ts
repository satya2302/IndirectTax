import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ReconcillationSynthetic {
  region: string;
  city: string;
  county: string;
  entity: string;
  gross: number;
  taxable: number;
  unreportedTax: number;
  taxRate: number;
  year: number;
  month: number;
}

@Injectable({ providedIn: 'root' })
export class ReconcillationSyntheticService {
  private apiUrl = 'http://localhost:5078/api/ReconcillationSynthetic';

  constructor(private http: HttpClient) {}

  getAll(): Observable<ReconcillationSynthetic[]> {
    return this.http.get<ReconcillationSynthetic[]>(this.apiUrl);
  }

  add(item: ReconcillationSynthetic): Observable<any> {
    return this.http.post(this.apiUrl, item);
  }

  addBulk(items: ReconcillationSynthetic[]): Observable<any> {
    return this.http.post(this.apiUrl, items);
  }
}
