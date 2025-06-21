import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface JournalSynthetic {
  region: string;
  city: string;
  county: string;
  entity: string;
  district: string;
  gross: number;
  taxable: number;
  inputSource: string;
  placeDetermination: string;
  entryId: number;
  id: string;
  logId: string;
  taxRate: number;
  year: number;
  month: number;
}

@Injectable({ providedIn: 'root' })
export class JournalSyntheticService {
  private apiUrl = 'http://localhost:5078/api/JournalSynthetic';

  constructor(private http: HttpClient) {}

  getAll(): Observable<JournalSynthetic[]> {
    return this.http.get<JournalSynthetic[]>(this.apiUrl);
  }

  addBulk(items: JournalSynthetic[]): Observable<any> {
    return this.http.post(this.apiUrl, items);
  }
}
