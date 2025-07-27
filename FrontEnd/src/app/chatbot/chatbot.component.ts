import { Component, EventEmitter, Output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { NgClass } from '@angular/common';
import { CommonModule } from '@angular/common';
import { ChatbotService, Message } from './chatbot.service';

@Component({
  selector: 'app-chatbot',
  standalone: true,
  imports: [CommonModule, FormsModule, NgClass],
  templateUrl: './chatbot.component.html',
  styleUrls: ['./chatbot.component.css']
})
export class ChatbotComponent {
  @Output() close = new EventEmitter<void>();

   constructor(private chatbotService : ChatbotService) {}

  messages: Array<{from: 'user'|'bot', text: string}> = [
    { from: 'bot', text: 'Hello! How can I help you today?' }
  ];
  userInput: string = '';

  sendMessage() {
    if (!this.userInput.trim()) return;
    this.messages.push({from: 'user', text: this.userInput});
    
    this.chatbotService.getUsage().subscribe({
      next: (data:Message[]) => {
        console.log(data)
        data.forEach(message  => {
          if (message.text.trim()) {
            this.messages.push({ from: 'bot', text:message.text });
          }
        });
        
      },
      error: (err) => {
        
      }
    });
    // Show API not ready message as markdown

    this.userInput = '';
  }

  onClose() {
    this.close.emit();
  }
}
