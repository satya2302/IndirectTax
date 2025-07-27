import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Message {
  from:string;
  text: string;

}

@Injectable({ providedIn: 'root' })
export class ChatbotService {
  private apiUrl = 'http://localhost:5000/chat';

  constructor(private http: HttpClient) {}

  getUsage(): Observable<Message[]> {
    return this.http.get<Message[]>(this.apiUrl);
  }
}
