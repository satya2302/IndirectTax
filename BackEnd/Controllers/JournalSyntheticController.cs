using Microsoft.AspNetCore.Mvc;
using IndirectTax.Data;
using IndirectTax.Models;
using System.Collections.Generic;

namespace IndirectTax.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class JournalSyntheticController : ControllerBase
    {
        private readonly JournalSyntheticRepository _repository;

        public JournalSyntheticController(JournalSyntheticRepository repository)
        {
            _repository = repository;
        }

        [HttpPost]
        public ActionResult<IEnumerable<JournalSynthetic>> Post([FromBody] IEnumerable<JournalSynthetic> items)
        {
            var data = _repository.BulkAddAndGetAll(items);
            return Ok(data);
        }

        [HttpGet]
        public ActionResult<IEnumerable<JournalSynthetic>> Get()
        {
            var data = _repository.GetAll();
            return Ok(data);
        }
    }
}
