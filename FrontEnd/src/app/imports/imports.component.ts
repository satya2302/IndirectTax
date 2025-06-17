import { Component } from '@angular/core';
import { ReconcillationSyntheticService, ReconcillationSynthetic } from './reconcillation-synthetic.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-imports',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './imports.component.html',
  styleUrl: './imports.component.css'
})
export class ImportsComponent {
  fileName = '';
  uploadError = '';
  uploadSuccess = false;
  loading = false;
  data: ReconcillationSynthetic[] = [];
  showUnreportedTaxOnly = false;
  filteredData: ReconcillationSynthetic[] = [];

  constructor(private readonly reconcillationService: ReconcillationSyntheticService) {}

  onFileSelected(event: any) {
    const file: File = event.target.files[0];
    if (file) {
      this.fileName = file.name;
      this.uploadError = '';
      this.uploadSuccess = false;
      this.readAndUploadCSV(file);
    }
  }

  readAndUploadCSV(file: File) {
    const reader = new FileReader();
    reader.onload = (e: any) => {
      const text = e.target.result;
      const rows = text.split(/\r?\n/).filter((row: string) => row.trim().length > 0);
      const dataRows = rows.slice(1);
      this.loading = true;
      const items: ReconcillationSynthetic[] = [];
      dataRows.forEach((row: string) => {
        const values = row.split(',');
        if (values.length < 10) { return; }
        items.push({
          region: values[0],
          city: values[1],
          county: values[2],
          entity: values[3],
          gross: +values[4],
          taxable: +values[5],
          unreportedTax: +values[6],
          taxRate: +values[7],
          year: +values[8],
          month: +values[9]
        });
      });
      if (items.length > 0) {
        this.reconcillationService.addBulk(items).subscribe({
          next: () => this.afterUpload(),
          error: () => {
            this.uploadError = 'Bulk upload failed.';
            this.loading = false;
          }
        });
      } else {
        this.uploadError = 'No valid data to upload.';
        this.loading = false;
      }
    };
    reader.readAsText(file);
  }

  afterUpload() {
    this.uploadSuccess = true;
    this.loading = false;
    this.fetchData();
    setTimeout(() => {
      this.fileName = '';
      this.uploadSuccess = false;
    }, 2000);
  }

  fetchData() {
    this.loading = true;
    this.reconcillationService.getAll().subscribe({
      next: (data) => {
        this.data = data;
        this.applyFilter();
        this.loading = false;
      },
      error: () => {
        this.uploadError = 'Failed to fetch data.';
        this.loading = false;
      }
    });
  }

  applyFilter() {
    if (this.showUnreportedTaxOnly) {
      this.filteredData = this.data.filter(row => row.unreportedTax > 0);
    } else {
      this.filteredData = this.data;
    }
  }

  onShowUnreportedTaxChange() {
    this.applyFilter();
  }

  get hasUnreportedTax(): boolean {
    return this.filteredData.some(row => row.unreportedTax > 0);
  }

  ngOnInit() {
    this.fetchData();
  }
}
