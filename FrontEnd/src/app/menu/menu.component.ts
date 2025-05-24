import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { UsageComponent } from '../usage/usage.component';
import { ImportsComponent } from '../imports/imports.component';
import { JournalsComponent } from '../journals/journals.component';
import { TaxReturnSummaryComponent } from '../taxreturnsummary/taxreturnsummary.component';

@Component({
  selector: 'app-menu',
  standalone: true,
  imports: [CommonModule, UsageComponent, ImportsComponent, JournalsComponent, TaxReturnSummaryComponent],
  templateUrl: './menu.component.html',
  styleUrl: './menu.component.css'
})
export class MenuComponent {
  activeTab: string = 'usage';

  navigate(tab: string) {
    this.activeTab = tab;
  }
}
