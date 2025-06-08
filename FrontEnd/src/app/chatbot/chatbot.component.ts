import { Component, EventEmitter, Output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { NgClass } from '@angular/common';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-chatbot',
  standalone: true,
  imports: [CommonModule, FormsModule, NgClass],
  templateUrl: './chatbot.component.html',
  styleUrls: ['./chatbot.component.css']
})
export class ChatbotComponent {
  @Output() close = new EventEmitter<void>();
  messages: Array<{from: 'user'|'bot', text: string}> = [
    { from: 'bot', text: 'Hello! How can I help you today?' }
  ];
  userInput: string = '';

  sendMessage() {
    if (!this.userInput.trim()) return;
    this.messages.push({from: 'user', text: this.userInput});
    // Show API not ready message as markdown
    setTimeout(() => {
      this.messages.push({from: 'bot', text: 'API is not ready.'});
    }, 500);
    this.userInput = '';
  }

  onClose() {
    this.close.emit();
  }
}
