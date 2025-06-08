import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { ChatbotComponent } from '../chatbot/chatbot.component';

@Component({
  selector: 'app-menu',
  standalone: true,
  imports: [CommonModule, RouterModule, ChatbotComponent],
  templateUrl: './menu.component.html',
  styleUrl: './menu.component.css'
})
export class MenuComponent {
  showChatbot = false;

  openChatbot(event: Event) {
    event.preventDefault();
    event.stopPropagation();
    this.showChatbot = true;
  }
}
