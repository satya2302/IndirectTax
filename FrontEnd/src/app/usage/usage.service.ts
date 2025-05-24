import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { DataUsage } from './data-usage.model';

@Injectable({ providedIn: 'root' })
export class UsageService {
  private apiUrl = 'http://localhost:5078/api/Usage';

  constructor(private http: HttpClient) {}

  getUsage(): Observable<DataUsage[]> {
    return this.http.get<DataUsage[]>(this.apiUrl);
  }
}
