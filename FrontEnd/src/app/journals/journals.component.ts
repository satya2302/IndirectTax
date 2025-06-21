import { Component, OnInit } from '@angular/core';
import { JournalSyntheticService, JournalSynthetic } from './journal-synthetic.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-journals',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './journals.component.html',
  styleUrl: './journals.component.css'
})
export class JournalsComponent implements OnInit {
  fileName = '';
  uploadError = '';
  uploadSuccess = false;
  uploadSuccessTimeout: any;
  loading = false;
  data: JournalSynthetic[] = [];
  filteredData: JournalSynthetic[] = [];
  page = 1;
  pageSize = 10;
  pagedData: JournalSynthetic[] = [];
  totalPages = 1;

  filters: { [key: string]: string } = {
    region: '',
    city: '',
    county: '',
    entity: '',
    district: '',
    gross: '',
    taxable: '',
    inputSource: '',
    placeDetermination: '',
    entryId: '',
    id: '',
    logId: '',
    taxRate: '',
    year: '',
    month: ''
  };

  constructor(private readonly journalService: JournalSyntheticService) {}

  ngOnInit() {
    this.journalService.getAll().subscribe({
      next: (res: JournalSynthetic[]) => {
        this.data = res;
        this.filteredData = res;
        this.updatePagedData();
      },
      error: (err) => {
        this.uploadError = 'Failed to load data.';
      }
    });
  }

  applyFilters() {
    this.filteredData = this.data.filter(row => {
      return Object.keys(this.filters).every(key => {
        const filterValue = this.filters[key].toLowerCase();
        if (!filterValue) return true;
        
        const cellValue = String(row[key as keyof JournalSynthetic] || '').toLowerCase();
        return cellValue.includes(filterValue);
      });
    });
    this.page = 1;
    this.updatePagedData();
  }

  updatePagedData() {
    this.totalPages = Math.ceil(this.filteredData.length / this.pageSize) || 1;
    const start = (this.page - 1) * this.pageSize;
    const end = start + this.pageSize;
    this.pagedData = this.filteredData.slice(start, end);
  }

  goToPage(page: number) {
    if (page < 1 || page > this.totalPages) return;
    this.page = page;
    this.updatePagedData();
  }

  nextPage() {
    this.goToPage(this.page + 1);
  }

  prevPage() {
    this.goToPage(this.page - 1);
  }

  onFilterChange() {
    this.applyFilters();
  }

  onFileSelected(event: any) {
    const file: File = event.target.files[0];
    if (file) {
      this.fileName = file.name;
      this.uploadError = '';
      this.uploadSuccess = false;
      if (this.uploadSuccessTimeout) {
        clearTimeout(this.uploadSuccessTimeout);
      }
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
      const items: JournalSynthetic[] = [];
      dataRows.forEach((row: string) => {
        const values = row.split(',');
        if (values.length < 15) { return; }
        items.push({
          region: values[0],
          city: values[1],
          county: values[2],
          entity: values[3],
          district: values[4],
          gross: +values[5],
          taxable: +values[6],
          inputSource: values[7],
          placeDetermination: values[8],
          entryId: +values[9],
          id: values[10],
          logId: values[11],
          taxRate: +values[12],
          year: +values[13],
          month: +values[14],
        });
      });
      this.journalService.addBulk(items).subscribe({
        next: (res: JournalSynthetic[]) => {
          this.data = res;
          this.filteredData = res;
          this.uploadSuccess = true;
          this.loading = false;
          this.uploadSuccessTimeout = setTimeout(() => {
            this.fileName = '';
            this.uploadSuccess = false;
          }, 2000);
        },
        error: (err) => {
          this.uploadError = 'Upload failed.';
          this.loading = false;
        }
      });
    };
    reader.readAsText(file);
  }
}
